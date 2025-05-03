from transformers import (
    GPT2Config, GPT2LMHeadModel, 
    LlamaConfig, LlamaForCausalLM, 
    GPTJConfig, GPTJForCausalLM,
    TrainerCallback, AutoModelForCausalLM,
    TrainerState
)
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import GPT2TokenizerFast
from torch.utils.data import Subset
from random import sample, seed
from pathlib import Path
import yaml
import argparse
import wandb
from gb_dataloader import GBDataset
import torch
import glob
import shutil
from torch.profiler import profile, record_function, ProfilerActivity



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default=None, help="Path to the training data")
    parser.add_argument("--eval_data_sets", type=str, nargs="+", default=[], help="Path to the evaluation data sets")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to the tokenizer")
    # model parameters
    parser.add_argument("--config", type=str, default="./config/llama-360M.yaml", help="Configuration file path")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed")
    # wandb arguments
    parser.add_argument("--use_wandb", type=bool, default=False, help="Use wandb for logging")
    parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    # continuous arguments
    parser.add_argument("--last_checkpoint", type=str, default=None, help="Path to the last checkpoint")
    # output
    parser.add_argument("--output_dir", type=str, default=None, help="Path to the output directory")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    def set_seed(random_seed: int):
        seed(random_seed)

        torch.manual_seed(random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)

    set_seed(args.random_seed)

    train_dataset = GBDataset(args.train_data, config['data']['seq_length'], random_chunk=False)

    if config['training'].get('gpus', None) is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = config['training']['gpus']

    print(f"using {config['training']['gpus']} GPUs")
    train_tokens = len(train_dataset) * config['data']['seq_length']
    print(f"train_tokens = {train_tokens/10**6}M")

    model_year = tuple(args.train_data.split("/")[-2].split("_"))

    all_eval_datasets = {}
    for data_set in args.eval_data_sets:
        full_eval_dataset = GBDataset(data_set, config['data']['seq_length'], offset=0)
        eval_samples = min(config['data']['eval_samples'], len(full_eval_dataset))
        eval_indices = sample(range(len(full_eval_dataset)), eval_samples)
        eval_dataset = Subset(full_eval_dataset, eval_indices)
        years = tuple(data_set.split("/")[-2].split("_"))
        all_eval_datasets[years] = eval_dataset

    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer_path)
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<pad>"
    tokenizer.model_max_length = config['data']['seq_length']

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    if args.last_checkpoint is None:
        if config['model']['type'] == "Llama":
            model_config = LlamaConfig(
                vocab_size=tokenizer.vocab_size,
                max_position_embeddings=2*tokenizer.model_max_length,
                hidden_size=config['model']['hidden_size'],
                intermediate_size=config['model']['intermediate_size'],
                num_hidden_layers=config['model']['n_layer'],
                num_attention_heads=config['model']['n_head'],
                num_key_value_heads=config['model'].get('n_KV', config['model']['n_head']),
                tie_word_embeddings=config['model'].get('tie_word_embeddings', False),
                pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
                attention_dropout=config['model'].get('attention_dropout', 0.0),
            )
            model = LlamaForCausalLM(model_config)
        elif config['model']['type'] == "GPT2":
            model_config = GPT2Config(
                vocab_size=tokenizer.vocab_size,
                n_positions=2*tokenizer.model_max_length,
                n_embd=config['model']['hidden_size'],
                n_layer=config['model']['n_layer'],
                n_head=config['model']['n_head'],
                resid_pdrop = config['model']['resid_pdrop'],
                embd_pdrop = config['model']['embd_pdrop'],
                attn_pdrop = config['model']['attn_pdrop'],
                pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
            )
            model = GPT2LMHeadModel(model_config)
        elif config['model']['type'] == "GPTJ":
            model_config = GPTJConfig(
                vocab_size=tokenizer.vocab_size,
                n_positions=2*tokenizer.model_max_length,
                n_embd=config['model']['hidden_size'],
                n_layer=config['model']['n_layer'],
                n_head=config['model']['n_head'],
                resid_pdrop = config['model']['resid_pdrop'],
                embd_pdrop = config['model']['embd_pdrop'],
                attn_pdrop = config['model']['attn_pdrop'],
                tie_word_embeddings=config['model']['tie_word_embeddings'],
                pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
            )
            model = GPTJForCausalLM(model_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.last_checkpoint
        )

    print(f'model parameters = {model.num_parameters()}')


    output_dir = args.output_dir
    accumulation_steps = config['training']['gradient_accumulation_steps']
    per_device_bsz = config['training']['batch_size'] // accumulation_steps

    total_steps = len(train_dataset) // (per_device_bsz * accumulation_steps)

    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"training length: {len(train_dataset)}")

    max_steps = total_steps
    if args.last_checkpoint is not None:
        trainer_state = TrainerState.load_from_json(os.path.join(args.last_checkpoint, "trainer_state.json"))
        max_steps += trainer_state.global_step
    
    training_args = TrainingArguments(
        max_steps=max_steps,
        output_dir=output_dir,
        gradient_accumulation_steps=accumulation_steps,
        per_device_train_batch_size=per_device_bsz,
        per_device_eval_batch_size=per_device_bsz,
        warmup_steps=config['training']['warmup_steps'], 
        lr_scheduler_type="cosine",
        learning_rate=float(config['training']['lr']),
        fp16=config['training']['fp16'],
        load_best_model_at_end=False,
        torch_compile = config['training'].get('torch_compile', False),
        logging_steps=20,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    

    if args.use_wandb:
        wandb.login()
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=config, id = args.wandb_name, resume="allow")

    if args.last_checkpoint is not None:
        trainer.train(resume_from_checkpoint=args.last_checkpoint)
    else:
        trainer.train()
    
    if args.use_wandb:
        try:
            prev_art = wandb.use_artifact("eval_results:latest")
            eval_table = prev_art.get("eval_results")
            print(f"Loaded {len(eval_table.data)} rows from previous artifact.")
        except wandb.errors.CommError:
            # First time: create a fresh Table
            columns = ["eval_start_year", "eval_end_year", "eval_loss", "model_start_year", "model_end_year", "model_name"]
            eval_table = wandb.Table(columns=columns)
        #print(eval_table)
    for eval_years, eval_dataset in all_eval_datasets.items():
        print(f"Evaluating on {eval_years} dataset")
        eval_results = trainer.evaluate(eval_dataset=eval_dataset)
        print(f"Evaluation results for {eval_years}: {eval_results}")
        if args.use_wandb:
            eval_table.add_data(
                eval_years[0], 
                eval_years[1], 
                eval_results['eval_loss'], 
                model_year[0],
                model_year[1],
                args.wandb_name
            )

    if args.use_wandb:
        #wandb.log({f"eval_results": eval_table}, step=trainer.state.global_step)
        new_art = wandb.Artifact(name="eval_results", type="evaluation_table")
        new_art.add(eval_table, "eval_results")
        wandb.log_artifact(new_art)

    trainer.save_model(output_dir)
    trainer.save_state()
    tokenizer.save_pretrained(output_dir)
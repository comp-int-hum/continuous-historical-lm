from transformers import (
    GPT2Config, GPT2LMHeadModel, 
    LlamaConfig, LlamaForCausalLM, 
    GPTJConfig, GPTJForCausalLM,
    TrainerCallback, AutoModelForCausalLM,
    TrainerState
)
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
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
    parser.add_argument("--eval_data_set", type=str, default=[], help="Path to the evaluation data sets")
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

    if args.use_wandb:
        wandb.login()
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=config)


    train_dataset = GBDataset(args.train_data, config['data']['seq_length'], random_chunk=True)

    if config['training'].get('gpus', None) is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = config['training']['gpus']

    print(f"using {config['training']['gpus']} GPUs")
    train_tokens = len(train_dataset) * config['data']['seq_length']
    print(f"train_tokens = {train_tokens/10**6}M")

    model_year = tuple(args.train_data.split("/")[-2].split("_"))

    full_eval_dataset = GBDataset(args.eval_data_set, config['data']['seq_length'], offset=0)
    eval_samples = min(config['data']['eval_samples'], len(full_eval_dataset))
    eval_indices = sample(range(len(full_eval_dataset)), eval_samples)
    eval_dataset = Subset(full_eval_dataset, eval_indices)

    print(f"eval_samples = {len(eval_dataset)}")

    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer_path)
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<pad>"
    tokenizer.model_max_length = config['data']['seq_length']

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    if args.last_checkpoint is not None:
        # Last checkpoint is the directory of the trainer state:
        trainer_state = TrainerState.load_from_json(os.path.join(args.last_checkpoint, "trainer_state.json"))
        # get best checkpoint from the trainer state
        args.last_checkpoint = trainer_state.best_model_checkpoint


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

    epochs_so_far = 0
    if args.last_checkpoint is not None:
        trainer_state = TrainerState.load_from_json(os.path.join(args.last_checkpoint, "trainer_state.json"))
        global_steps = trainer_state.global_step
        current_epoch = global_steps // total_steps
        epochs_so_far = current_epoch
        print(f"trainer state global steps: {trainer_state.global_step}")
        trainer_state.best_metric = float('inf')
        trainer_state.save_to_json(os.path.join(args.last_checkpoint, "trainer_state.json"))

    max_epoch = epochs_so_far + config['training']['num_epochs']
    print(f"Total training steps: {total_steps}")
    print(f"per device batch size: {per_device_bsz}")
    print(f"accumulation steps: {accumulation_steps}")
    print(f"epochs so far: {epochs_so_far}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        gradient_accumulation_steps=accumulation_steps,
        per_device_train_batch_size=per_device_bsz,
        per_device_eval_batch_size=per_device_bsz,
        report_to="wandb",
        warmup_steps=config['training']['warmup_steps'], 
        lr_scheduler_type="cosine",
        learning_rate=float(config['training']['lr']),
        fp16=config['training']['fp16'],
        weight_decay=float(config['training']['weight_decay']),
        logging_steps=20,
        save_total_limit=2,
        num_train_epochs=max_epoch,
        ignore_data_skip   = True,
        save_strategy= "epoch", 
        evaluation_strategy= "epoch",
        logging_strategy= "steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        torch_compile = config['training'].get('torch_compile', False),
    )
    early_stop = EarlyStoppingCallback(
        early_stopping_patience   = 1,
        early_stopping_threshold  = 0.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[early_stop],
    )
    


    if args.last_checkpoint is not None:
        trainer.train(resume_from_checkpoint=args.last_checkpoint)
    else:
        trainer.train()

    trainer.save_model(output_dir)
    trainer.save_state()
    tokenizer.save_pretrained(output_dir)
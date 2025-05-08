from transformers import (
    GPT2TokenizerFast,
    LlamaForCausalLM,
    LlamaConfig,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    TrainerState,
    EarlyStoppingCallback
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  Subset
from random import sample
import yaml
import torch
from random import seed

from pathlib import Path
import wandb
import argparse
from gb_dataloader import GBDataset
import gc
from distill_utils import DistillationTrainer, DistillationTrainingArguments
import glob
import shutil
import os
from transformers import TrainerCallback, TrainerControl



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", help="Path to the training data")
    parser.add_argument("--eval_data_set", help="Path to the evaluation data")
    parser.add_argument("--tokenizer_path", help="Path to the tokenizer")
    # teacher models
    parser.add_argument("--teacher_dir_1", help="Path to the first teacher model")
    parser.add_argument("--teacher_dir_2", help="Path to the second teacher model")
    # model parameters
    parser.add_argument("--config", type=str, default="./config/llama-16M.yaml", help="Configuration file path")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed")
    # wandb arguments
    parser.add_argument("--use_wandb", type=bool, default=False, help="Use wandb for logging")
    parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    # continuous arguments
    parser.add_argument("--last_checkpoint", type=str, default=None, help="Path to the last checkpoint")
    # output
    parser.add_argument("--output_dir", help="Path to the output directory")
    args, rest = parser.parse_known_args()


    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if config['training'].get('gpus', None) is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = config['training']['gpus']

    def set_seed(random_seed: int):
        seed(random_seed)

        torch.manual_seed(random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)

    set_seed(args.random_seed)
        
    tokenizer_path = args.tokenizer_path
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<pad>"
    tokenizer.model_max_length = config['data']['seq_length']

    model_year = tuple(args.train_data.split("/")[-2].split("_"))

    if args.lr:
        config['training']['lr'] = args.lr

    # Dynamic Model Configuration
    if args.last_checkpoint is None:
        print("Creating student model from config")
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
                attention_dropout=config['model'].get('attention_dropout', 0.0)
            )
            student = LlamaForCausalLM(model_config)
        else:
            raise ValueError(f"Model type {config['model']['type']} not supported for student model")
    else:
        print(f"Loading student model from {args.last_checkpoint}")
        student = AutoModelForCausalLM.from_pretrained(
            args.last_checkpoint,
        )
    print("Student device:", student.device)
    print(student.device)
    train_dataset = GBDataset(args.train_data, config['data']['seq_length'], random_chunk=True)
    token_count = len(train_dataset) * config['data']['seq_length']

    full_eval_dataset = GBDataset(args.eval_data_set, config['data']['seq_length'], offset=0)
    eval_samples = min(config['data']['eval_samples'], len(full_eval_dataset))
    eval_indices = sample(range(len(full_eval_dataset)), eval_samples)
    eval_dataset = Subset(full_eval_dataset, eval_indices)

    teacher1_path = args.teacher_dir_1
    teacher2_path = args.teacher_dir_2

    teacher1 = LlamaForCausalLM.from_pretrained(args.teacher_dir_1, torch_dtype=torch.float16)
    teacher2 = LlamaForCausalLM.from_pretrained(args.teacher_dir_2, torch_dtype=torch.float16)
    teachers = [teacher1, teacher2]
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )


    print(f'model num parameters: student = {student.num_parameters()}')
    print(f'model num parameters: teacher1 = {teacher1.num_parameters()}')
    print(f'model num parameters: teacher2 = {teacher2.num_parameters()}')


    if args.use_wandb:
        wandb.login()
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=config)


    output_dir = args.output_dir
    accumulation_steps = config['training']['gradient_accumulation_steps']
    per_device_bsz = config['training']['batch_size'] // accumulation_steps

    total_steps = len(train_dataset) // (per_device_bsz * accumulation_steps)

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


    training_args = DistillationTrainingArguments(
        output_dir=args.output_dir,
        gradient_accumulation_steps=accumulation_steps,
        per_device_train_batch_size=per_device_bsz,
        report_to="wandb",
        warmup_steps=config['training']['warmup_steps'], 
        lr_scheduler_type="cosine",
        learning_rate=float(config['training']['lr']),
        fp16=config['training']['fp16'],
        weight_decay=float(config['training']['weight_decay']),
        alpha=float(config['training']['alpha']),
        temperature=float(config['training']['temperature']),
        logging_steps=20,
        save_total_limit=2,
        num_train_epochs=max_epoch,
        ignore_data_skip   = True,
        save_strategy= "epoch", 
        evaluation_strategy= "epoch",
        logging_strategy= "steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    early_stop = EarlyStoppingCallback(
        early_stopping_patience   = 1,
        early_stopping_threshold  = 0.0,
    )

    trainer = DistillationTrainer(
            student,
            training_args,
            teacher_models=teachers,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[early_stop],
        )

    if args.last_checkpoint is not None:
        print(f"Loading trainer from {args.last_checkpoint}")
        trainer.train(resume_from_checkpoint=args.last_checkpoint)
    else:
        print("Starting training from scratch")
        trainer.train()


    trainer.save_model(args.output_dir)
    trainer.save_state()
    tokenizer.save_pretrained(args.output_dir)
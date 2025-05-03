from transformers import (
    GPT2TokenizerFast,
    LlamaForCausalLM,
    LlamaConfig,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    TrainerState
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
import argparse
from gb_dataloader import GBDataset
import gc
from distill_utils import DistillationTrainer, DistillationTrainingArguments
import glob
import shutil
import os
from transformers import TrainerCallback, TrainerControl
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_data_sets", nargs="+", help="Path to the evaluation data")
    parser.add_argument("--tokenizer_path", help="Path to the tokenizer")
    # teacher models
    parser.add_argument("--config", type=str, default="./config/llama-16M.yaml", help="Configuration file path")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed")
    # output
    parser.add_argument("--model", type=str, default=None, help="Path to the last checkpoint")
    parser.add_argument("--output", help="Path to the output file")
    args, rest = parser.parse_known_args()


    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

    model_year = tuple(args.model.split("/")[-2].split("_"))

    if args.lr:
        config['training']['lr'] = args.lr

    # Dynamic Model Configuration
    model = AutoModelForCausalLM.from_pretrained(
            args.model,
        )

    all_eval_datasets = {}
    print(args.eval_data_sets)
    for data_set in args.eval_data_sets:
        full_eval_dataset = GBDataset(data_set, config['data']['seq_length'], offset=0)
        eval_samples = min(config['data']['eval_samples'], len(full_eval_dataset))
        eval_indices = sample(range(len(full_eval_dataset)), eval_samples)
        eval_dataset = Subset(full_eval_dataset, eval_indices)
        years = tuple(data_set.split("/")[-2].split("_"))
        all_eval_datasets[years] = eval_dataset


    accumulation_steps = config['training']['gradient_accumulation_steps']
    per_device_bsz = config['training']['batch_size'] // accumulation_steps

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    eval_path = os.path.dirname(args.output.replace(".jsonl", ""))

    training_args = TrainingArguments(
        output_dir=eval_path,
        do_train=False,
        do_eval=False,
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
    )

    
    for eval_years, eval_dataset in all_eval_datasets.items():
        print(f"Evaluating on {eval_years} dataset")
        eval_results = trainer.evaluate(eval_dataset=eval_dataset)
        print(f"Evaluation results for {eval_years}: {eval_results}")
        # write to jsonl file args.output
        with open(args.output, 'a') as f:
            temp = {
                "eval_start_year": eval_years[0],
                "eval_end_year": eval_years[1],
                "eval_loss": eval_results['eval_loss'],
                "model_start_year": model_year[0],
                "model_end_year": model_year[1],
            }
            f.write(json.dumps(temp) + "\n")
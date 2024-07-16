import argparse
import os
import sys
from typing import Tuple

import hydra
import numpy as np
import torch
import wandb
import yaml
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from trl import SFTTrainer
from transformers import AutoTokenizer, TrainingArguments, EarlyStoppingCallback
from peft import LoraConfig
from contextlib import nullcontext
from huggingface_hub.hf_api import HfFolder
from omegaconf import DictConfig

from src.util import get_small_dataset, save_args_to_hf, PROJECT_DIR, get_dataset

"""
Run:
python src/model/sft_model.py --config_path configs --config_name sft
"""

parser = argparse.ArgumentParser(description="This script fine tunes a model with SFT.")
parser.add_argument(
    "--config_path",
    type=str,
    required=True,
)
parser.add_argument(
    "--config_name",
    type=str,
    required=True,
)
parser.add_argument(
    "--add_config_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--add_config_name",
    type=str,
    default=None,
)

args, unknown = parser.parse_known_args()
# Construct hydra compatible sys.argv
sys.argv = ['main.py']

# torch.set_default_device('cuda')


def get_trainer(cfg: DictConfig):
    dataset_train, dataset_dev = get_dataset(cfg.dataset_size, cfg.dataset_name, cfg.language)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=True)

    training_args = TrainingArguments(
        "stojchet/" + model_full_name(),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        push_to_hub=True,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        logging_steps=100,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        weight_decay=cfg.weight_decay if cfg.weight_decay is not None else 0,
        num_train_epochs=cfg.epochs,
        report_to="wandb",
        eval_strategy="steps",
        eval_steps=100,
        warmup_steps=200,
        lr_scheduler_type="linear",
        load_best_model_at_end=True,
    )

    kwargs = {'revision': 'main', 'trust_remote_code': False, 'attn_implementation': None, 'torch_dtype': None,
              'use_cache': False, 'device_map': None, 'quantization_config': None}

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=5,
        early_stopping_threshold=0.0
    )

    def formatting_func(example):
        return "```python" + example[cfg.dataset_ref_field] + "```"

    model_name = model_full_name().split("/")[-1]
    run = wandb.init(
        project="huggingface",
        entity="stojchets",
        id=model_name,
    )

    init_context = nullcontext()
    with init_context:
        trainer = SFTTrainer(
            base_model,
            train_dataset=dataset_train,
            eval_dataset=dataset_dev,
            dataset_text_field=cfg.dataset_ref_field,
            max_seq_length=cfg.max_seq_length,
            dataset_batch_size=16,
            args=training_args,
            model_init_kwargs=kwargs,
            peft_config=LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                bias="none",
                # task_type='CAUSAL_LM',
                # use_rslora=True,
            ) if not cfg.no_lora else None,
            tokenizer=tokenizer,
            packing=False,
            callbacks=[early_stopping_callback],

            # formatting_func=formatting_func,
        )

    return trainer


def model_full_name():
    if args.add_config_path is None:
        return args.config_name
    else:
        return args.add_config_name + "-" + args.config_name


@hydra.main(config_path=str(PROJECT_DIR / args.config_path), config_name=args.config_name)
def main(cfg: DictConfig):
    load_dotenv()

    HfFolder.save_token(os.getenv('HF_WRITE_TOKEN'))
    global base_model
    base_model = cfg.base_model if args.add_config_path is None else "stojchet/" + args.add_config_name

    trainer = get_trainer(cfg)
    trainer.train()
    trainer.push_to_hub()
    save_args_to_hf(dict(cfg), "stojchet/" + model_full_name(), "model")
    if args.add_config_path is not None:
        data = yaml.safe_load(open(PROJECT_DIR / f"{args.add_config_path}/{args.add_config_name}.yaml", "r"))
        save_args_to_hf(data, "stojchet/" + model_full_name(), "model", path_in_repo="coarse_params.json")


if __name__ == "__main__":
    main()

import argparse
import os
import sys
from typing import Tuple

import hydra
import numpy as np
import wandb
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from omegaconf import DictConfig
from trl import KTOConfig, KTOTrainer
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, EarlyStoppingCallback
from peft import LoraConfig
from contextlib import nullcontext
from huggingface_hub.hf_api import HfFolder
import torch

from src.util import get_small_dataset, save_args_to_hf, PROJECT_DIR, get_dataset

"""
This script is used to train a KTO model

Run:
python3 kto _model.py --config_path configs --config_name kto
"""

parser = argparse.ArgumentParser(description="This script fine tunes a model with KTO.")
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

args, unknown = parser.parse_known_args()

# Construct hydra compatible sys.argv
sys.argv = ['main.py']


def get_trainer(cfg: DictConfig):
    """
    This function is used to prepare the trainer for KTO.

    Parameter:
    DictConfig - that contains the hyperparameters of the model
    """
    dataset_train, dataset_dev = get_dataset(cfg.dataset_size if cfg.dataset_size != "inf" else np.inf, cfg.dataset_name, cfg.language)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    model_args = KTOConfig(
        "stojchet/" + model_full_name(),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        push_to_hub=True,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        logging_steps=100,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        lr_scheduler_type="linear",
        eval_steps=100,
        warmup_ratio=0.1,
        # bf16=True,
        fp16=True,
        eval_strategy="steps",
        num_train_epochs=cfg.epochs,
        warmup_steps=200,
        load_best_model_at_end=True,
        weight_decay=cfg.weight_decay if cfg.weight_decay is not None else 0,
    )
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, quantization_config=quantization_config)
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=5,
        early_stopping_threshold=0.0
    )
    run = wandb.init(
        project="huggingface",
        entity="stojchets",
        id=model_full_name(),
    )

    init_context = nullcontext()
    with init_context:
        dpo_trainer = KTOTrainer(
            cfg.base_model,
            train_dataset=dataset_train,
            eval_dataset=dataset_dev,
            tokenizer=tokenizer,
            args=model_args,
            peft_config=LoraConfig(  # type: ignore
                r=64,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                # task_type='CAUSAL_LM',
                target_modules="all-linear"
            ) if not cfg.no_lora else None,
            callbacks=[early_stopping_callback],

        )

    return dpo_trainer


def model_full_name():
    return args.config_name


@hydra.main(config_path=str(PROJECT_DIR / args.config_path), config_name=args.config_name)
def main(cfg: DictConfig):
    load_dotenv()

    HfFolder.save_token(os.getenv('HF_WRITE_TOKEN'))
    trainer = get_trainer(cfg)
    trainer.train()
    trainer.push_to_hub()
    save_args_to_hf(dict(cfg), "stojchet/" + model_full_name(), "model")


if __name__ == "__main__":
    main()

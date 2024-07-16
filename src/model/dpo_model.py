import argparse
import os
import sys

import hydra
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig
from trl import DPOTrainer
from transformers import AutoTokenizer, TrainingArguments, EarlyStoppingCallback
from peft import LoraConfig
from contextlib import nullcontext
from huggingface_hub.hf_api import HfFolder

from src.util import save_args_to_hf, get_dataset, PROJECT_DIR

"""
Run:
python3 dpo_model.py --config_path configs --config_name dpo
"""

parser = argparse.ArgumentParser(description="This script fine tunes a model with DPO.")
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
    dataset_train, dataset_dev = get_dataset(cfg.dataset_size, cfg.dataset_name, cfg.language)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)

    model_args = TrainingArguments(
        "stojchet/" + model_full_name(),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        push_to_hub=True,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        logging_steps=100, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        num_train_epochs=cfg.epochs,
        eval_strategy="steps",
        eval_steps=100,
        warmup_steps=200,
        lr_scheduler_type="linear",
        load_best_model_at_end=True,
        weight_decay=cfg.weight_decay if cfg.weight_decay is not None else 0,
        # bf16=True,
        fp16=True,
    )

    model_name = model_full_name().split("/")[-1]
    run = wandb.init(
        project="huggingface",
        entity="stojchets",
        id=model_name,
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=5,
        early_stopping_threshold=0.0
    )
    init_context = nullcontext()

    with init_context:
        dpo_trainer = DPOTrainer(
            cfg.base_model,
            beta=0.1,
            train_dataset=dataset_train,
            eval_dataset=dataset_dev,
            tokenizer=tokenizer,
            args=model_args,
            peft_config=LoraConfig(  # type: ignore
                r=64,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                task_type='CAUSAL_LM',
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

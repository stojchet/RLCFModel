import argparse
from typing import Tuple

import numpy as np
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers import AutoTokenizer, TrainingArguments
from peft import LoraConfig
from contextlib import nullcontext

from src.model.util import get_small_dataset

"""
Run:
python src/model/sft_model.py --max_seq_length 512 --model_name_or_path deepseek-ai/deepseek-coder-1.3b-base
--dataset_size 10000 
"""

parser = argparse.ArgumentParser(description="This script fine tunes a model with SFT.")
parser.add_argument(
    "--per_device_train_batch_size",
    type=int,
    default=16,
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=16,
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=1.41e-5,
)
parser.add_argument(
    "--max_seq_length",
    type=int,
    default=512,
)
parser.add_argument(
    "--lora_r",
    type=int,
    default=64,
)
parser.add_argument(
    "--lora_alpha",
    type=int,
    default=16,
)
parser.add_argument(
    "--lora_dropout",
    type=float,
    default=0.05,
)
parser.add_argument(
    "--dataset_size",
    type=int,
    default=np.inf,
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default="code_search_net",
)
parser.add_argument(
    "--base_model",
    type=str,
    default="deepseek-ai/deepseek-coder-1.3b-base",
)
parser.add_argument(
    "--out_model",
    type=str,
    required=True,
)
parser.add_argument(
    "--dataset_ref_field",
    type=str,
    default="whole_func_string",
)


def get_dataset(dataset_size: int, hf_dataset_path: str) -> Tuple[Dataset, Dataset]:
    dataset_train = load_dataset(hf_dataset_path, split="train", streaming=True if dataset_size != np.inf else False)
    dataset_dev = load_dataset(hf_dataset_path, split="validation", streaming=True if dataset_size != np.inf else False)
    if dataset_size != np.inf:
        dataset_train = get_small_dataset(dataset_train, dataset_size)
        dataset_dev = get_small_dataset(dataset_dev, dataset_size)
    return dataset_train, dataset_dev


def get_trainer(args: argparse.Namespace):
    dataset_train, dataset_dev = get_dataset(args.dataset_size, args.dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True)

    training_args = TrainingArguments(args.out_model, per_device_train_batch_size=args.per_device_train_batch_size,
                                      push_to_hub=True,
                                      gradient_accumulation_steps=args.gradient_accumulation_steps,
                                      learning_rate=args.learning_rate,
                                      logging_steps=1.0, gradient_checkpointing=True,
                                      gradient_checkpointing_kwargs={'use_reentrant': False})
    kwargs = {'revision': 'main', 'trust_remote_code': False, 'attn_implementation': None, 'torch_dtype': None,
              'use_cache': False, 'device_map': None, 'quantization_config': None}

    init_context = nullcontext()
    with init_context:
        trainer = SFTTrainer(
            args.base_model,
            train_dataset=dataset_train,
            eval_dataset=dataset_dev,
            dataset_text_field=args.dataset_ref_field,
            max_seq_length=args.max_seq_length,
            dataset_batch_size=16,
            args=training_args,
            model_init_kwargs=kwargs,
            peft_config=LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type='CAUSAL_LM',
            ),
            tokenizer=tokenizer,
            packing=False,
        )

    return trainer


if __name__ == "__main__":
    args = parser.parse_args()

    trainer = get_trainer(args)
    trainer.train()
    trainer.push_to_hub()

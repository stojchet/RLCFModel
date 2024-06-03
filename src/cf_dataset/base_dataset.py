import argparse
import os
from typing import Optional

import numpy as np
from datasets import load_dataset
import torch
from tqdm import tqdm
from datasets import Dataset
import gc
from dotenv import load_dotenv

from util import get_small_dataset, dtype_from_string
from src.model_impl import Model

PREDICTION = "prediction"
PROMPT = "func_documentation_string"
LABEL = "func_code_string"

"""
Example command run:
python3 base_dataset.py \
--hf_ds_out_path=base_dataset \
--language=java \
--dataset_name=stojchet/csn_filtered_subset \
--prompt=. \
--torch_dtype=float16
"""

parser = argparse.ArgumentParser(description="This script fine tunes a model with SFT.")
parser.add_argument(
    "--language",
    type=str,
    required=True,
)
parser.add_argument(
    "--hf_ds_out_path",
    type=str,
    required=True,
    help="Repo id in HF where the base dataset will be saved"
)
parser.add_argument(
    "--prompt_path",
    type=str,
    required=True,
    help="Path to the file containing system prompt"
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default="code_search_net",
)
parser.add_argument(
    "--model_name",
    type=str,
    default="deepseek-ai/deepseek-coder-1.3b-base",
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=2000,
)
parser.add_argument(
    "--torch_dtype",
    type=str,
    default="bfloat16",
)
parser.add_argument(
    "--save_intermediate",
    type=bool,
    default=True,
    help="If true, saves the dataset at every new 100 datapoints, under revision i/100."
)
parser.add_argument(
    "--dataset_size",
    type=int,
    default=np.inf,
)

torch.set_default_device("cuda")
torch.set_float32_matmul_precision('high')
load_dotenv()


def _load_dataset(dataset_name: str, language: str, dataset_size: int) -> Dataset:
    dataset = load_dataset(dataset_name, split="train", trust_remote_code=True)

    if language != "all":
        dataset = dataset.filter(lambda x: x["language"] == language)
    if dataset_size != np.inf:
        dataset = get_small_dataset(dataset.to_iterable_dataset(), dataset_size)

    return dataset


def _collect_predictions(
        dataset: Dataset,
        model: Model,
        save_intermediate: bool,
        hf_ds_out_path: str,
        base_prompt: str,
) -> Dataset:
    final_dataset = []
    for i, datapoint in tqdm(enumerate(dataset)):
        datapoint[PREDICTION] = model.predict(base_prompt + "\n" + datapoint[PROMPT])
        torch.cuda.empty_cache()
        gc.collect()
        final_dataset.append(datapoint)

        if save_intermediate and (i + 1) % 1000 == 0:
            ds = Dataset.from_list(final_dataset)
            _save_dataset(ds, hf_ds_out_path, str(float((i + 1) / 2)))
            print(i + 1)

    return Dataset.from_list(final_dataset)


def _save_dataset(dataset: Dataset, hf_ds_out_path: str, revision: Optional[str] = None,
                  language: Optional[str] = None) -> None:
    dataset.push_to_hub(
        hf_ds_out_path,
        token=os.getenv('HF_WRITE_TOKEN'),
        revision=revision,
        config_name=language if language else "default"
    )


def create_dataset(args: argparse.Namespace) -> None:
    model = Model(name=args.model_name,
                  max_new_tokens=args.max_new_tokens,
                  torch_dtype=dtype_from_string(args.torch_dtype),
                  truncation=True)
    dataset = _load_dataset(dataset_name=args.dataset_name,
                            language=args.language,
                            dataset_size=args.dataset_size)

    with open(args.prompt_path) as prompt_file:
        prompt = prompt_file.read()

    predictions = _collect_predictions(dataset, model, args.save_intermediate,
                                       args.hf_ds_out_path + "_" + args.language, prompt)
    _save_dataset(predictions, args.hf_ds_out_path, revision="main", language=args.language)


if __name__ == "__main__":
    args = parser.parse_args()
    create_dataset(args)

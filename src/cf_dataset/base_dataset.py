import argparse
import ast
import os
import re
import sys
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
from datasets import load_dataset
import torch
from huggingface_hub import HfApi, HfFolder
from omegaconf import DictConfig
from tqdm import tqdm
from datasets import Dataset
import gc
from dotenv import load_dotenv

from cf_dataset.code_string_util import extract_func_def_and_docstring
from src.util import get_small_dataset, dtype_from_string, PROJECT_DIR
from src.model_impl import Model

PREDICTION = "prediction"
PROMPT = "func_documentation_string"
FULL_CODE = "whole_func_string"
LABEL = "func_code_string"

"""
Example command run:
python3 base_dataset.py \
--language=java \
--dataset_name=stojchet/csn_filtered_subset \
--config_path configs/base
--config_name deepseek_bs1
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
    "--language",
    type=str,
    required=True,
)
parser.add_argument(
    "--prompt_path",
    type=str,
    required=True,
    help="Path to the file containing system prompt"
)

args, unknown = parser.parse_known_args()

# Construct hydra compatible sys.argv
sys.argv = ['main.py']

torch.set_default_device("cuda")
torch.set_float32_matmul_precision('high')


def _load_dataset(dataset_name: str, language: str, dataset_size: int, split: str) -> Dataset:
    dataset = load_dataset(dataset_name, split=split, name=language, trust_remote_code=True)
    if dataset_size != np.inf:
        dataset = get_small_dataset(dataset.to_iterable_dataset(), dataset_size)

    return dataset


def get_out_ds_name(model_name: str, prompt_name: str) -> str:
    return model_name.split("/")[-1] + "-" + prompt_name


def prepare_prompt(base_prompt: str, datapoint):
    if datapoint["language"] == "python":
        func_def, fdef_docstring = extract_func_def_and_docstring(datapoint[FULL_CODE])
        return ("", "") if fdef_docstring == None else (func_def, base_prompt + fdef_docstring)
    else:
        raise NotImplementedError("No support for java func extraction yet")


def get_batches_from_dataset(dataset, batch_size):
    return [dataset.iloc[i:i + batch_size] for i in range(0, len(dataset), batch_size)]


def _collect_predictions(
        dataset: Dataset,
        model: Model,
        prompt_name: str,
        base_prompt: str,
        batch_size: int,
        split: str
) -> Dataset:
    final_dataset = []
    batches = get_batches_from_dataset(dataset.to_pandas(), batch_size)
    for i, batch in tqdm(enumerate(batches)):
        batch_results = [res for _, datapoint in batch.iterrows() if
                         (res := prepare_prompt(base_prompt, datapoint)) is not None]

        func_defs, full_prompts = zip(*batch_results)
        batch[PREDICTION] = model.predict(list(full_prompts))
        # prepared prompt is what is being sent to the model for generation
        # Note this is not 100% accurate - indentation is added by hand -> rely on ast for extracting function body
        batch["prepared_prompt"] = list(full_prompts)
        batch["func_def"] = list(func_defs)
        torch.cuda.empty_cache()
        gc.collect()
        final_dataset += batch.to_dict("records")

        save_intermediate_ds(final_dataset, (i + 1) * batch_size, model, prompt_name, split)

    return Dataset.from_list(final_dataset)


def save_intermediate_ds(final_dataset, i, model, prompt_name, split, n=1000):
    if i % n == 0:
        ds = Dataset.from_list(final_dataset)
        _save_dataset(ds, "temp-" + get_out_ds_name(model.name, prompt_name), split=split, revision=str(float(i / n)))
        print(i)


def _save_dataset(dataset: Dataset, hf_ds_out_path: str, split: str,
                  revision: Optional[str] = None,
                  language: Optional[str] = None) -> None:
    dataset.push_to_hub(
        hf_ds_out_path,
        token=os.getenv('HF_WRITE_TOKEN'),
        revision=revision,
        config_name=language if language else "default",
        split=split
    )


def _upload_prompt():
    HfApi().upload_file(
        path_or_fileobj=PROJECT_DIR / args.config_path / (args.config_name + ".yaml"),
        path_in_repo="prompt.txt",
        repo_id="stojchet/" + get_out_ds_name(args.config_name, Path(args.prompt_path).stem),
        repo_type="dataset"
    )


@hydra.main(config_path=str(PROJECT_DIR / args.config_path), config_name=args.config_name)
def main(cfg: DictConfig) -> None:
    model = Model(name=cfg.model_name,
                  max_new_tokens=cfg.max_new_tokens,
                  torch_dtype=dtype_from_string(cfg.torch_dtype),
                  truncation=True)

    for split in ["train", "validation", "test"]:
        dataset = _load_dataset(dataset_name=cfg.dataset_name,
                                language=args.language,
                                dataset_size=cfg.dataset_size if cfg.dataset_size != "inf" else np.inf,
                                split=split)

        with open(PROJECT_DIR / args.prompt_path) as prompt_file:
            prompt = prompt_file.read()

        predictions = _collect_predictions(dataset, model, Path(args.prompt_path).stem, prompt, cfg.batch_size, split)
        _save_dataset(predictions, "stojchet/" + get_out_ds_name(args.config_name, Path(args.prompt_path).stem),
                      split=split,
                      revision="main", language=args.language)

    _upload_prompt()


if __name__ == "__main__":
    load_dotenv()
    HfFolder.save_token(os.getenv('HF_WRITE_TOKEN'))
    main()

import argparse
import os
import sys
from pathlib import Path

import hydra
import numpy as np
from datasets import load_dataset
import torch
from huggingface_hub import HfApi, HfFolder
from omegaconf import DictConfig
from datasets import Dataset
from dotenv import load_dotenv

from src.cf_dataset.util import get_out_ds_name, _collect_predictions, _save_dataset
from src.util import get_small_dataset, dtype_from_string, PROJECT_DIR
from src.model_impl import Model

PREDICTION = "prediction"

"""
This script collects the base dataset. It takes the seed dataset and collects predictions and uploads in on HuggingFace.

Example command run:
python3 base_dataset.py \
--language=java \
--prompt_path=stojchet/csn_filtered_subset \
--config_path configs/base
--config_name deepseek_bs1
"""

parser = argparse.ArgumentParser(description="This script collects a base dataset with specified models' predictions.")
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
# todo: either have all langauge setups in yaml files or as args
parser.add_argument(
    "--language",
    type=str,
    required=True,
)
parser.add_argument(
    "--prompt_path",
    type=str,
    default=None,
    help="Path to the file containing system prompt"
)

args, unknown = parser.parse_known_args()

# Construct hydra compatible sys.argv
sys.argv = ['main.py']
#
torch.set_default_device("cuda")
# torch.set_float32_matmul_precision('high')


def _load_dataset(dataset_name: str, language: str, dataset_size: int, split: str) -> Dataset:
    dataset = load_dataset(dataset_name, split=split, name=language, trust_remote_code=True)
    if dataset_size != np.inf:
        dataset = get_small_dataset(dataset.to_iterable_dataset(), dataset_size)

    return dataset


def _upload_prompt():
    HfApi().upload_file(
        path_or_fileobj=PROJECT_DIR / args.config_path / (args.config_name + ".yaml"),
        path_in_repo="prompt.txt",
        repo_id="stojchet/" + get_out_ds_name(args.config_name, Path(args.prompt_path).stem if args.prompt_path is not None else "empty"),
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
        # dataset = get_small_dataset(dataset.to_iterable_dataset(), 100)

        if args.prompt_path is not None:
            with open(PROJECT_DIR / args.prompt_path) as prompt_file:
                prompt = prompt_file.read()
        else:
            prompt = ""

        predictions = _collect_predictions(dataset, model, prompt, cfg.batch_size, split, args.language)
        # for now dataset name if config name + prompt
        _save_dataset(predictions, "stojchet/" + get_out_ds_name(args.config_name, Path(args.prompt_path).stem if args.prompt_path is not None else "empty"),
                      split=split,
                      revision="main",
                      language=args.language)

    _upload_prompt()


if __name__ == "__main__":
    load_dotenv()
    HfFolder.save_token(os.getenv('HF_WRITE_TOKEN'))
    main()

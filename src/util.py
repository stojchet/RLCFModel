import argparse
import hashlib
import json
import tempfile
from functools import partial
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from datasets import Dataset, IterableDataset, load_dataset
from huggingface_hub import HfApi

PROJECT_DIR = Path(__file__).parent.parent

def get_small_dataset(dataset: IterableDataset, n: int = 100) -> Dataset:
    dataset = dataset.take(n)

    def gen_from_iterable_dataset(iterable_ds):
        yield from iterable_ds

    dataset = Dataset.from_generator(partial(gen_from_iterable_dataset, dataset), features=dataset.features)
    return dataset

def get_dataset_excluding_first_n(dataset: IterableDataset, n: int = 200) -> Dataset:
    dataset = dataset.skip(n)

    def gen_from_iterable_dataset(iterable_ds):
        yield from iterable_ds

    dataset = Dataset.from_generator(partial(gen_from_iterable_dataset, dataset), features=dataset.features)
    return dataset


def get_dataset(dataset_size: int, hf_dataset_path: str, language: str) -> Tuple[Dataset, Dataset]:
    dataset_train = load_dataset(hf_dataset_path, split="train", name=language, streaming=True if dataset_size != np.inf else False)
    dataset_dev = load_dataset(hf_dataset_path, split="validation", name=language, streaming=True if dataset_size != np.inf else False)
    if dataset_size != np.inf:
        dataset_train = get_small_dataset(dataset_train, dataset_size)
        dataset_dev = get_small_dataset(dataset_dev, 2000)
    return dataset_train, dataset_dev


def dtype_from_string(dtype_str):
    dtype = getattr(torch, dtype_str)
    return dtype


def save_args_to_hf(args: Dict, dataset_name: str, repo_type: str, path_in_repo: str = "params.json"):
    # upload hyperparameter config
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as temp:
        json.dump(args, temp)
        temp_path = temp.name

    HfApi().upload_file(
        path_or_fileobj=temp_path, path_in_repo=path_in_repo, repo_id=dataset_name, repo_type=repo_type
    )


# make this the dataset name
def hash_dict(dictionary):
    json_dict = json.dumps(dictionary, sort_keys=True)
    return hashlib.md5(json_dict.encode('utf-8')).hexdigest()


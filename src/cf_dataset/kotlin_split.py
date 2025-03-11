import argparse
import os
from typing import Optional

from datasets import load_dataset, Dataset


parser = argparse.ArgumentParser(description="Split kotlin dataset in train, validation and test set.")
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
    "--prompt_path",
    type=str,
    default=None,
    help="Path to the file containing system prompt"
)


def _load_dataset(dataset_name: str, split: str) -> Dataset:
    return load_dataset(dataset_name, split=split, name="kotlin", trust_remote_code=True)

def _save_dataset(dataset: Dataset, hf_ds_out_path: str, split: str,
                  revision: Optional[str] = None) -> None:
    print("Saving dataset to {} {} {}".format(hf_ds_out_path, revision, "kotlin"))
    dataset.push_to_hub(
        hf_ds_out_path,
        token=os.getenv('HF_WRITE_TOKEN'),
        revision=revision,
        config_name="kotlin",
        split=split
    )


def get_out_ds_name() -> str:
    return "stojchet/base_prediction_dataset"


args, unknown = parser.parse_known_args()

dataset = _load_dataset(dataset_name="stojchet/base_prediction_dataset",
                        split="train")

_save_dataset(dataset, get_out_ds_name() + "temp",
              split="train",
              revision="main")

split_dataset = dataset.train_test_split(test_size=0.15)
temp_train_dataset = split_dataset['train']
test_dataset = split_dataset['test']

final_split = temp_train_dataset.train_test_split(test_size=0.11)

train_dataset = final_split['train']
validation_dataset = final_split['test']

_save_dataset(train_dataset, get_out_ds_name(),
              split="train",
              revision="main")

_save_dataset(validation_dataset, get_out_ds_name(),
              split="validation",
              revision="main")

_save_dataset(test_dataset, get_out_ds_name(),
              split="test",
              revision="main")
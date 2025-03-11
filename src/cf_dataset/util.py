import gc
import os
from typing import Dict, Optional

import torch
from pyarrow._dataset import Dataset
from tqdm import tqdm

from src.model_impl import Model

DOCUMENTATION = "documentation"
CODE_COMPLETION = "code_completion"
FUNCTION_DEFINITION = "function_def"
LANGUAGE = "language"

def get_whole_reference_function(datapoint: Dict[str, str]) -> str:
    return datapoint[FUNCTION_DEFINITION] + "\n" + datapoint[CODE_COMPLETION]

def get_whole_prediction_function(datapoint: Dict[str, str], code_completion: str) -> str:
    return datapoint[FUNCTION_DEFINITION] + "\n" + code_completion


"""
Utils for base_dataset
"""


import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from datasets import Dataset
import gc
from src.model_impl import Model

PREDICTION = "prediction"

# todo: add language to the name
def get_out_ds_name(config_name, prompt_path) -> str:
    return get_out_ds_name_args(config_name, prompt_path)


def get_out_ds_name_args(config_name, prompt_name) -> str:
    return config_name + "-" + prompt_name


# todo: can I do this somehow better?
def get_batches_from_dataset(dataset, batch_size):
    return [dataset.iloc[i:i + batch_size] for i in range(0, len(dataset), batch_size)]


def _collect_predictions(
        dataset: Dataset,
        model: Model,
        base_prompt: str,
        batch_size: int,
        split: str,
        language: str,
) -> Dataset:
    """
    This function collects the model predictions on the dataset in batches, and prepares a final dataset.

    Parameters:
    dataset (Dataset): The input dataset on which model's prediction is to be run.
    model (Model): The model object used for making predictions.
    base_prompt (str): The base prompt string which is used to prepare a full prompt with a data point from the batch.
    batch_size (int): The size of the batch that needs to be prepared from the dataset.
    split (str): Split type, which could be 'train', 'test' or 'valid'.

    Returns:
    Dataset: The final_dataset consists of the batch data along with model's predictions,
             prepared prompts and function definitions.
    """
    device = torch.device("cuda")

    # Move the model to the device
    model.model.to(device)

    final_dataset = []
    batches = get_batches_from_dataset(dataset.to_pandas(), batch_size)
    for i, batch in tqdm(enumerate(batches)):
        full_prompts = [base_prompt + datapoint["prompt"] for _, datapoint in batch.iterrows()]

        batch[PREDICTION] = [*model.predict(full_prompts)]
        # batch[PREDICTION] = [""]
        # prepared prompt is what is being sent to the model for generation
        # Note this is not 100% accurate - indentation is added by hand -> rely on ast for extracting function body
        torch.cuda.empty_cache()
        gc.collect()
        final_dataset += batch.to_dict("records")

        save_intermediate_ds(final_dataset, (i + 1) * batch_size, split, language=language)

    return Dataset.from_list(final_dataset)


def save_intermediate_ds(final_dataset, i, split, language, n=1000):
    """
    This function stores intermediate dataset to the Hub at revision `dataset size / n` for every n generated predictions
    """
    if i % n == 0:
        ds = Dataset.from_list(final_dataset)
        _save_dataset(ds, "stojchet/" + get_out_ds_name(), split=split, revision=str(float(i / n)), language=language)
        print(i)


def _save_dataset(dataset: Dataset, hf_ds_out_path: str, split: str, language: str,
                  revision: Optional[str] = None) -> None:
    """
    This function saves the provided Dataset object to the Hub.

    Parameters:
    dataset (Dataset): The Dataset object to be saved.
    hf_ds_out_path (str): The Hub path where the dataset is to be saved.
    split (str): The split for the dataset. Could be "train", "test", "valid".
    revision (Optional[str]): The revision string for version control.
    This parameter is used for intermediate storing of the dataset.

    """
    print("Saving dataset to {} {} {}".format(hf_ds_out_path, revision, language))
    dataset.push_to_hub(
        hf_ds_out_path,
        token=os.getenv('HF_WRITE_TOKEN'),
        revision=revision,
        config_name=language,
        split=split
    )

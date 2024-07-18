import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import hydra
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfFolder
from omegaconf import DictConfig

from src.cf_dataset.compiler import compile_function
from src.evaluate.sanitize.code_block_extraction import sanitize_dataset
from src.util import PROJECT_DIR

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["PYTHONBREAKPOINT"] = "0"

parser = argparse.ArgumentParser(description="Sanitize and evaluate predictions.")
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
parser.add_argument(
    "--prompt_path",
    type=str,
    required=True,
)
parser.add_argument(
    "--pred_dataset",
    type=str,
    default=None,
)
parser.add_argument(
    "--language",
    type=str,
    default=None,
)
parser.add_argument(
    "--out",
    type=str,
    default=None,
)


args, unknown = parser.parse_known_args()

# Construct hydra compatible sys.argv
sys.argv = ['main.py']


def get_dataset(dataset_name: str, name: str):
    """
    This function loads a given dataset, sanitizes it, prints the number of examples that compile successfully,
    and saves it to a temporary file.
    The goal is to prepare the dataset on which mxeval will be run.

    Parameters:
    dataset_name (str): The name or path of the dataset to load.
    name (str): The name used for the loaded dataset.

    Returns:
    str: The path of the temporary file.

    """
    dataset = load_dataset("stojchet/" + dataset_name, revision="main", split="train", name=name, trust_remote_code=True)
    sanitized = sanitize_dataset(dataset, "", predictions_field="completion")

    print_number_of_examples_that_compile(sanitized, name)

    ds_name_dir = dataset_name.split("/")[-1] + "-" + name
    dirpath = PROJECT_DIR / "src/evaluate/temp" / ds_name_dir
    if not dirpath.exists(): os.mkdir(dirpath)
    temp_filepath = PROJECT_DIR / f"src/evaluate/temp/{ds_name_dir}/predictions.jsonl"
    with open(temp_filepath, "w") as temp_file:
        for datapoint in sanitized:
            temp_file.write(json.dumps(datapoint) + "\n")

    return temp_filepath


def print_number_of_examples_that_compile(sanitized, benchmark):
    """
    This function prints the number of function completions in a sanitized dataset that compile.

    Parameters:
    sanitized (list): The sanitized dataset, which is a list of dictionaries where each dictionary contains a completion.
    benchmark (str): The name of the benchmark or dataset being processed.
    """
    i = 0
    for example in sanitized:
        if compile_function[args.language](example["prompt"] + "\n" + example["completion"]) and len(example["completion"]) > 0:
            i += 1
    print(f"Number of examples that compile in {benchmark}: {i}")


def dataset_full_name():
    return f"{args.language}-{model_full_name().split('/')[-1]}-{Path(args.prompt_path).stem}"


def model_full_name():
    if args.add_config_path is None:
        return "stojchet/" + args.config_name
    else:
        return "stojchet/" + args.add_config_name + "-" + args.config_name


def log_mxeval_results_on_wandb(project, value):
    run = wandb.init(project=project, entity="stojchets")
    wandb.log({"mbxp": value})
    run.finish()


def log(dataset_name: str, value: float):
    run = wandb.init(entity="stojchets", project="huggingface", id=dataset_name, resume="must")
    wandb.log({f"mbxp-{Path(args.prompt_path).stem}": value})
    run.finish()


def init_wandb(model_name):
    wandb.init(
        project="huggingface",
        entity="stojchets",
        id=model_name,
    )


@hydra.main(config_path=str(PROJECT_DIR / args.config_path), config_name=args.config_name)
def main(cfg: DictConfig):
    load_dotenv()
    HfFolder.save_token(os.getenv('HF_WRITE_TOKEN'))

    args.language = args.language if args.pred_dataset else cfg["language"]
    dataset_name = args.pred_dataset.split("/")[-1] if args.pred_dataset else dataset_full_name()
    print(dataset_name)

    mbxp_temp = get_dataset(dataset_name, "mbxp")
    humaneval_temp = get_dataset(dataset_name, "humaneval")
    if args.language == "python":
        mbxp_data = PROJECT_DIR / 'external/mxeval/data/mbxp/mbpp_release_v1.jsonl'
        humaneval_data = PROJECT_DIR / "external/mxeval/data/multilingual_humaneval/HumanEval.jsonl"
    else:
        mbxp_data = PROJECT_DIR / 'external/mxeval/data/mbxp/mbjp_release_v1.2.jsonl'
        humaneval_data = PROJECT_DIR / "external/mxeval/data/multilingual_humaneval/HumanEval_java_v1.1.jsonl"

    command = [PROJECT_DIR / 'external/mxeval/mxeval/evaluate_functional_correctness.py', mbxp_temp,
               '--problem_file', mbxp_data]
    subprocess.call(command)

    command = [PROJECT_DIR / 'external/mxeval/mxeval/evaluate_functional_correctness.py', humaneval_temp,
               '--problem_file', humaneval_data]
    subprocess.call(command)


if __name__ == "__main__":
    main()

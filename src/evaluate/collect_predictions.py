import argparse
import gc
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Dict

import hydra
import pandas as pd
import torch
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import HfFolder, HfApi
from omegaconf import DictConfig
from tqdm import tqdm
import locale

from src.model_impl import Model
from src.util import dtype_from_string, PROJECT_DIR
from external.mxeval.mxeval.data import read_problems, ROOT, MULTILINGUAL_HUMANEVAL_METADATA, \
    MULTILINGUAL_MBXP_METADATA

locale.getpreferredencoding = lambda: "UTF-8"

"""
Download and setup mxeval in `external` directory

Run:
python3 src/evaluate/collect_predictions.py \
--config_path=configs \
--config_name=sft \
--path_to_prompt=../prompts/simple.txt \
--batch_size=8
"""

parser = argparse.ArgumentParser(description="This script evaluates a pre trained model using EvalPlus.")
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
    "--model",
    type=str,
    default=None,
)
parser.add_argument(
    "--language",
    type=str,
    default=None,
)

parser.add_argument(
    "--path_to_prompt",
    type=str,
    required=True,
    help="Path to the prompt that will be used. Prompt is provided in file."
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=4,
)

parser.add_argument(
    "--torch_dtype",
    type=str,
    default="float16",
    help="Path to the prompt that will be used. Prompt is provided in file."
)

args, unknown = parser.parse_known_args()

# Construct hydra compatible sys.argv
sys.argv = ['main.py']

if torch.cuda.is_available():
    torch.set_default_device('cuda')
else:
    raise ValueError("No GPU is available!")


def get_model(model_name: str, torch_dtype: torch.dtype):
    return Model(model_name, torch_dtype, max_new_tokens=1000)


def get_prompt_from_file(path_to_prompt: str):
    with open(PROJECT_DIR / path_to_prompt, "r") as file:
        return file.read().strip()


def get_humaneval_lang_dataset(lang: str):
    return os.path.join(ROOT, "..", "data", "multilingual_humaneval", MULTILINGUAL_HUMANEVAL_METADATA[lang])


def get_mbxp_lang_dataset(lang: str):
    return os.path.join(ROOT, "..", "data", "mbxp", MULTILINGUAL_MBXP_METADATA[lang])


def get_predictions_mxeval(model: Model, prompt: str, problems: str, batch_size: int) -> List[Dict[str, str]]:
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp:
        temp.write(json.dumps(problems) + '\n')

    dataset = pd.read_json(temp.name).transpose()
    bathes = get_batches_from_dataset(dataset, batch_size)

    num_samples_per_task = 1
    samples = [
        sample
        for batch in tqdm(bathes)
        for sample in generate_batch(model, batch, prompt)
        for _ in range(num_samples_per_task)
    ]
    return samples


def get_batches_from_dataset(dataset, batch_size):
    return [dataset.iloc[i:i + batch_size] for i in range(0, len(dataset), batch_size)]


def generate_batch(model, problems_subset, prompt):
    predictions = model.predict([prompt + task["prompt"] for idx, task in problems_subset.iterrows()])
    return [{
        "task_id": idx,
        "language": val["language"],
        "completion": predictions[i],
        "prompt": val["prompt"],
    } for i, (idx, val) in enumerate(problems_subset.iterrows())]


def run_eval(args: argparse.Namespace, model_name):
    model = get_model(model_name, dtype_from_string(args.torch_dtype))
    prompt = get_prompt_from_file(args.path_to_prompt)

    mbxp_problems = read_problems(get_mbxp_lang_dataset(args.language))
    predictions = get_predictions_mxeval(model, prompt, mbxp_problems, args.batch_size)
    print("mbxp predictions collected")
    dataset = Dataset.from_list(predictions)
    publish_dataset(dataset, "mbxp")

    humaneval_problems = read_problems(get_humaneval_lang_dataset(args.language))
    predictions = get_predictions_mxeval(model, prompt, humaneval_problems, args.batch_size)
    print("humaneval predictions collected")
    dataset = Dataset.from_list(predictions)
    publish_dataset(dataset, "humaneval")


def publish_dataset(dataset, benchmark):
    dataset_name = dataset_full_name()
    print(dataset_name)

    dataset.push_to_hub(dataset_name, config_name=benchmark)
    HfApi().upload_file(
        path_or_fileobj=PROJECT_DIR / args.path_to_prompt, path_in_repo="prompt.txt", repo_id=dataset_name,
        repo_type="dataset"
    )


def dataset_full_name():
    return f"stojchet/{args.language}-{model_full_name().split('/')[-1]}-{Path(args.path_to_prompt).stem}"


def model_full_name():
    if args.model is None:
        if args.add_config_path is None:
            return "stojchet/" + args.config_name
        else:
            return "stojchet/" + args.add_config_name + "-" + args.config_name
    else:
        return "stojchet/" + args.model.split("/")[-1]


@hydra.main(config_path=str(PROJECT_DIR / args.config_path), config_name=args.config_name)
def main(cfg: DictConfig) -> None:
    torch.cuda.empty_cache()
    gc.collect()

    load_dotenv()
    HfFolder.save_token(os.getenv('HF_WRITE_TOKEN'))

    if args.model is not None:
        model_name = args.model
    else:
        model_name = model_full_name()
        args.language = cfg.language

    run_eval(args, model_name)


if __name__ == "__main__":
    main()

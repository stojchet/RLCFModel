import argparse
import gc
import os
import random
import time
from pathlib import Path
from typing import List, Dict

import torch
from datasets import Dataset
from dotenv import load_dotenv
from evalplus.data import get_mbpp_plus, write_jsonl
from huggingface_hub import HfFolder, HfApi
from tqdm import tqdm
import locale

from src.evaluate.util import get_split_name
from src.model_impl import Model
from src.util import dtype_from_string
from external.mxeval.mxeval.data import write_jsonl, read_problems, ROOT, MULTILINGUAL_HUMANEVAL_METADATA, \
    MULTILINGUAL_MBXP_METADATA

locale.getpreferredencoding = lambda: "UTF-8"

"""
Download and setup mxeval in `external` directory

Run:
python3 src/evaluate/collect_predictions.py \
--predictions_file_name=predictions.jsonl \
--model_name=deepseek-ai/deepseek-coder-6.7b-instruct \
--language=python \
--path_to_prompt=../prompts/simple.txt
"""

parser = argparse.ArgumentParser(description="This script evaluates a pre trained model using EvalPlus.")
parser.add_argument(
    "--predictions_file_name",
    type=str,
    required=True,
    help="Name of the file where predictions will be stored"
)
parser.add_argument(
    "--model_name",
    type=str,
    required=True,
)
parser.add_argument(
    "--language",
    type=str,
    required=True,
)
parser.add_argument(
    "--framework",
    type=str,
    required=True,
)
parser.add_argument(
    "--path_to_prompt",
    type=str,
    required=True,
    help="Path to the prompt that will be used. Prompt is provided in file."
)
parser.add_argument(
    "--torch_dtype",
    type=str,
    default=torch.float32,
    help="Path to the prompt that will be used. Prompt is provided in file."
)

torch.set_default_device('cuda')


def get_model(model_name: str, torch_dtype: torch.dtype):
    return Model(model_name, torch_dtype)


def get_prompt_from_file(path_to_prompt: str):
    with open(path_to_prompt, "r") as file:
        return file.read().strip()


def get_predictions_evalplus(model: Model, prompt: str, language: str) -> List[Dict[str, str]]:
    return [
        {
            "task_id": task_id,
            "solution": model.predict(prompt + problem["prompt"]),
            "prompt": problem["prompt"],
            "language": language
        }
        for task_id, problem in tqdm(list(get_mbpp_plus().items()))
    ]


def get_humaneval_lang_dataset(lang: str):
    return os.path.join(ROOT, "..", "data", "multilingual_humaneval", MULTILINGUAL_HUMANEVAL_METADATA[lang])


def get_mbxp_lang_dataset(lang: str):
    return os.path.join(ROOT, "..", "data", "mbxp", MULTILINGUAL_MBXP_METADATA[lang])


def get_predictions_mxeval(model: Model, prompt: str, language: str) -> List[Dict[str, str]]:
    problems = read_problems(get_mbxp_lang_dataset(language))
    problems = dict(random.sample(problems.items(), 50))
    num_samples_per_task = 1
    start = time.time()
    samples = [
        generate_single(model, problems, prompt, task_id)
        for task_id in problems
        for _ in range(num_samples_per_task)
    ]
    print(time.time() - start)
    device = next(model.model.parameters()).device
    print(device)
    return samples


def generate_single(model, problems, prompt, task_id):
    return {
        "task_id": task_id,
        "language": problems[task_id]["language"],
        "completion": model.predict(prompt + problems[task_id]["prompt"]),
        "prompt": problems[task_id]["prompt"],
    }


def run_eval(args: argparse.Namespace, get_predictions):
    model = get_model(args.model_name, dtype_from_string(args.torch_dtype))
    prompt = get_prompt_from_file(args.path_to_prompt)
    predictions = get_predictions(model, prompt, args.language)
    write_jsonl(args.predictions_file_name, predictions)

    dataset = Dataset.from_list(predictions)
    publish_dataset(args, dataset)


def publish_dataset(args, dataset):
    dataset_name = f"stojchet/{get_split_name(args.model_name, args.language, 'mbpp', Path(args.path_to_prompt).stem)}"
    dataset.push_to_hub(dataset_name)
    HfApi().upload_file(
        path_or_fileobj=args.path_to_prompt, path_in_repo="prompt.txt", repo_id=dataset_name, repo_type="dataset"
    )


if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()

    load_dotenv()
    HfFolder.save_token(os.getenv('HF_WRITE_TOKEN'))
    args = parser.parse_args()

    if args.framework == "mxeval":
        run_eval(args, get_predictions_mxeval)
    elif args.framework == "evalp":
        run_eval(args, get_predictions_evalplus)

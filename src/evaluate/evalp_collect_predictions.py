import argparse

import torch
from evalplus.data import get_mbpp_plus, write_jsonl

from src.model_impl import Model

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


def dtype_from_string(dtype_str):
    dtype = getattr(torch, dtype_str)
    return dtype


def get_model(model_name: str, torch_dtype: torch.dtype):
    return Model(model_name, torch_dtype)


def get_prompt_from_file(path_to_prompt: str):
    with open(path_to_prompt, "r") as file:
        return file.read().strip()


def get_predictions(model: Model, prompt: str):
    return [
        dict(task_id=task_id, solution=model.predict(prompt + problem["prompt"]))
        for task_id, problem in list(get_mbpp_plus().items())
    ]


def run_eval(args: argparse.Namespace):
    model = get_model(args.model_name, args.torch_dtype)
    prompt = get_prompt_from_file(args.path_to_prompt)
    predictions = get_predictions(model, prompt)
    write_jsonl(args.predictions_file_name, predictions)

    import locale
    locale.getpreferredencoding = lambda: "UTF-8"


if __name__ == "__main__":
    args = parser.parse_args()
    run_eval(args)

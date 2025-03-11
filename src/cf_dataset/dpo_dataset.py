import argparse
import os

from datasets import Dataset, load_dataset
from tqdm import tqdm

from src.cf_dataset.corrupt_code import corrupt_code
from src.cf_dataset.compiler import compile_function
from src.cf_dataset.dataset_constants import LABEL, PREDICTION
from src.cf_dataset.util import get_whole_prediction_function
from src.evaluate.sanitize.function_extraction import extract_function_completion
from src.util import get_small_dataset

"""
This script takes a base dataset and creates a KTO compatible dataset

Run:
python3 dpo_dataset.py \
--base_dataset_name stojchet/temp-deepseek-coder-1.3b-base-python_markdown \
--language python 

"""

def get_args():
    parser = argparse.ArgumentParser(description="Creates a dataset for KTO -> positive/negative examples dataset.")
    parser.add_argument(
        "--base_dataset_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--language",
        type=str,
        default="all",
    )
    parser.add_argument(
        "--not_compile",
        type=float,
        default=1.0,
        help="For DPO this represents the size as percentage of the original dataset. "
             "If not enough examples produced by the model don't compile, the predictions are corrupted."
             "If set to 0 it will not corrupt anything"
        # todo: add better explanation
    )

    return parser.parse_args()

"""
Note: this will remove all comments inside the function body also
"""


def create_dpo_dataset(
        dataset: Dataset,
        language: str,
        not_compile: float,
        prompt_const: str = "prompt",
        target_const: str = LABEL,
        prediction_const: str = PREDICTION
) -> Dataset:
    """
    This function creates a new dataset for Direct Preference Optimization (DPO).
    For each data point in the base dataset, extracts the function completion from the prediction.
    If the function does not compile, it is added to the new dataset as a 'rejected' item
    along with the 'prompt' and 'chosen' item.

    Parameters:
    dataset (Dataset): The input dataset as a huggingface Dataset object.
    language (str): The language (For e.g: "python", "java") of the functions in dataset.
    prompt_const (str): The constant string to identify the prompts in the dataset.
    target_const (str): The constant string to identify the targets in the dataset.
    prediction_const (str): The constant string to identify the predictions in the dataset.

    Returns:
    Dataset
    """

    # add all examples that don't compile as false examples
    # ps paper proved that it's ok to have a dis-balance of positive negative examples
    nonempty = 0
    all_compile = 0

    new_dataset = []

    corrupt_predictions = []
    corrupt_count = not_compile * len(dataset)

    for datapoint in tqdm(dataset):
        prediction = datapoint[prediction_const].replace("<｜begin▁of▁sentence｜>", "") # todo: remove
        prediction = prediction.replace("<｜end▁of▁sentence｜>", "")

        # If the model continues generating tokens after completing the function those should be removed
        prediction_body = extract_function_completion(prediction, datapoint[prompt_const])

        prompt = datapoint[prompt_const]

        whole_prediction_func = get_whole_prediction_function(datapoint, prediction_body)
        reference_body = datapoint["code_completion"]

        # if whole prediction function doesn't compile
        # we know that whole ref function compiles because it's filtered out at the beginning
        # todo: that's a mistake it can be used here and in kto dataset
        if not compile_function[language](whole_prediction_func):
            new_dataset.append(
                {
                    "prompt": prompt,
                    "rejected": prediction_body,
                    "chosen": reference_body,
                }
            )
            corrupt_count -= 1
        else:
            all_compile += 1

            # todo: maybe concat the solution that compiles as chosen?

            corrupt_predictions.append(
                {
                    "prompt": prompt,
                    "rejected": corrupt_code(prediction_body, datapoint["language"]),
                    "chosen": reference_body,
                }
            )

        if prediction_body != "":
            nonempty += 1

    print("nonempty: " + str(nonempty))
    print("compile: " + str(all_compile))
    print("corrupt: " + str(corrupt_count))
    new_dataset += corrupt_predictions[:corrupt_count]

    return Dataset.from_list(new_dataset)


def get_base_dataset(dataset_name: str, language: str, split: str) -> Dataset:
    return load_dataset(dataset_name, split=split, name=language)


if __name__ == "__main__":
    args = get_args()
    for split in ["train", "validation", "test"]:
        print(split)

        base_dataset = get_base_dataset(args.base_dataset_name, args.language, split)
        # base_dataset = get_small_dataset(base_dataset.to_iterable_dataset(), 10)
        out_dataset = create_dpo_dataset(base_dataset, args.language, args.not_compile).shuffle()
        print("len: " + str(len(out_dataset)))

        prefix = "" if args.corrupt == 0.0 else "corrupted-"
        out_dataset.push_to_hub(
            f"stojchet/{prefix}dpo-" + args.base_dataset_name.split("/")[-1],
            token=os.getenv('HF_WRITE_TOKEN'),
            revision="main",
            config_name=args.language,
            split=split
        )

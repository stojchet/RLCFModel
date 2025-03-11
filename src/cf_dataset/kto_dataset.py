import argparse
import os

from datasets import Dataset, load_dataset
from tqdm import tqdm

from src.cf_dataset.compiler import compile_function
from src.cf_dataset.corrupt_code import corrupt_code
from src.cf_dataset.dataset_constants import LABEL, PREDICTION
from src.cf_dataset.util import get_whole_prediction_function
from src.evaluate.sanitize.function_extraction import extract_function_completion
from src.util import get_small_dataset

"""
This script takes a base dataset and creates a KTO compatible dataset

Run:
python3 kto_dataset.py \
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
        default=0.5,
        help="Percentage of examples in the final dataset that do not compile."
             "If not enough bad model predictions, the predictions will be corrupted."
    )

    return parser.parse_args()

def create_kto_dataset(
        dataset: Dataset,
        language: str,
        not_compile: float,
        prompt_const: str = "prompt",
        target_const: str = LABEL,
        prediction_const: str = PREDICTION
) -> Dataset:
    """
    This function creates a new dataset for Kahneman-Tversky Optimization (KTO).
    For each data point, it extracts the function completion from the prediction.
    If the function does not compile, it is added to the new dataset with a 'label' set to False.

    Parameters:
    dataset (Dataset): The input dataset as a huggingface Dataset object.
    language (str): The language (For e.g: "python", "java") of the functions in dataset.
    prompt_const (str): The constant string to identify the prompts in the dataset.
    target_const (str): The constant string to identify the targets in the dataset.
    prediction_const (str): The constant string to identify the predictions in the dataset.

    """
    # add all examples that don't compile as false examples
    # p.s. paper proved that it's ok to have a dis-balance of positive negative examples
    nonempty = 0
    all_compile = 0

    not_compile_dataset = []
    corrupt_dataset = []

    for datapoint in tqdm(dataset):
        prediction = datapoint[prediction_const].replace("<｜begin▁of▁sentence｜>", "") # todo: remove
        prediction = prediction.replace("<｜end▁of▁sentence｜>", "")
        code = extract_function_completion(prediction, datapoint[prompt_const])

        prompt = datapoint[prompt_const]

        whole_prediction_func = get_whole_prediction_function(datapoint, code)
        if not compile_function[language](whole_prediction_func):
            not_compile_dataset.append(
                {
                    "prompt": prompt,
                    "completion": code,
                    "label": False,
                }
            )
        else:
            all_compile += 1

            corrupt_dataset.append(
                {
                    "prompt": prompt,
                    "completion": corrupt_code(code, language),
                    "label": False,
                }
            )
        if code != "":
            nonempty += 1


    # total_not_compile_count / (len(dataset) + total_not_compile_count) = not_compile
    total_not_compile_count = int(len(dataset) * not_compile / (1 - not_compile))
    corrupt_examples_count = max(0, total_not_compile_count - len(not_compile_dataset))

    new_dataset = not_compile_dataset + corrupt_dataset[:corrupt_examples_count]

    print("nonempty: " + str(nonempty))
    print("compile: " + str(all_compile))

    # add all labels as true examples
    # prompt is prepared prompt -> what is sent to LM
    new_dataset += [
        {
            "prompt": datapoint[prompt_const],
            "completion": datapoint["code_completion"],
            "label": True,
        }
        for datapoint in dataset
    ]

    return Dataset.from_list(new_dataset)


def get_base_dataset(dataset_name: str, language: str, split: str) -> Dataset:
    return load_dataset(dataset_name, split=split, name=language)


if __name__ == "__main__":
    args = get_args()
    print(args)
    for split in ["train",  "validation", "test"]:
        print(split)
        base_dataset = get_base_dataset(args.base_dataset_name, args.language, split)
        # base_dataset = get_small_dataset(base_dataset.to_iterable_dataset(), 10)

        out_dataset = create_kto_dataset(base_dataset, args.language, not_compile=args.not_compile).shuffle()
        print("len: " + str(len(out_dataset)))

        out_dataset.push_to_hub(
            "stojchet/kto-" + args.base_dataset_name.split("/")[-1],
            token=os.getenv('HF_WRITE_TOKEN'),
            revision="main",
            config_name=args.language,
            split=split
        )

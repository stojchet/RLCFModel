import argparse
import os

from datasets import Dataset, load_dataset
from tqdm import tqdm

from src.cf_dataset.code_string_util import extract_function_body, extract_func_def_and_docstring
from src.cf_dataset.compiler import compile_function
from src.cf_dataset.dataset_constants import LABEL, PREDICTION
from src.evaluate.sanitize.function_extraction import extract_function_completion
from src.util import get_small_dataset

"""
This script takes a base dataset and creates a KTO compatible dataset

Run:
python3 kto_dataset.py \
--base_dataset_name stojchet/temp-deepseek-coder-1.3b-base-python_markdown \
--language python 

"""

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


def create_kto_dataset(
        dataset: Dataset,
        language: str,
        prompt_const: str,
        target_const: str = LABEL,
        prediction_const: str = PREDICTION
) -> Dataset:
    new_dataset = []

    # add all examples that don't compile as false examples
    # ps paper proved that it's ok to have a dis-balance of positive negative examples
    nonempty = 0
    all_compile = 0

    for datapoint in tqdm(dataset):
        prediction = datapoint[prediction_const].replace("<｜begin▁of▁sentence｜>", "")
        prediction = prediction.replace("<｜end▁of▁sentence｜>", "")
        code = extract_function_completion(prediction, datapoint[prompt_const])

        if language == "java":
            prompt = extract_func_def_and_docstring(datapoint, language)[1]
            code += "\n}"
        else:
            prompt = datapoint[prompt_const]

        whole_func = prompt + "\n" + code
        if not compile_function[language](whole_func) and code != "":
            new_dataset.append(
                {
                    "prompt": prompt,
                    "completion": code,
                    "label": False,
                }
            )
        else:
            all_compile += 1
        if code != "":
            nonempty += 1

    print("nonempty: " + str(nonempty))
    print("compile: " + str(all_compile))

    # add all labels as true examples
    # prompt is prepared prompt -> what is sent to LM
    new_dataset += [
        {
            "prompt": datapoint[prompt_const],
            "completion": extract_function_body(datapoint[target_const], language),
            "label": True,
        }
        for datapoint in get_small_dataset(dataset.to_iterable_dataset(), len(new_dataset))
    ]

    return Dataset.from_list(new_dataset)


def get_base_dataset(dataset_name: str, language: str, split: str) -> Dataset:
    return load_dataset(dataset_name, split=split, name=language)


if __name__ == "__main__":
    args = parser.parse_args()
    for split in ["train",  "validation", "test"]:
        print(split)
        base_dataset = get_base_dataset(args.base_dataset_name, args.language, split)
        out_dataset = create_kto_dataset(base_dataset, args.language, "prepared_prompt").shuffle()
        print("len: " + str(len(out_dataset)))

        out_dataset.push_to_hub(
            "stojchet/kto-" + args.base_dataset_name.split("/")[-1],
            token=os.getenv('HF_WRITE_TOKEN'),
            revision="main",
            config_name=args.language,
            split=split
        )

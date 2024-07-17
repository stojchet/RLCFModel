import argparse
import ast
import os

from datasets import Dataset, load_dataset
from tqdm import tqdm

from cf_dataset.corrupt_code import corrupt_python_code, corrupt_java_code
from src.cf_dataset.code_string_util import extract_function_body, extract_func_def_and_docstring
from src.cf_dataset.compiler import compile_function
from src.cf_dataset.dataset_constants import LABEL, PREDICTION
from src.evaluate.sanitize.function_extraction import extract_function_completion

"""
This script takes a base dataset and creates a KTO compatible dataset

Run:
python3 dpo_dataset.py \
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
parser.add_argument(
    "--corrupt",
    type=int,
    default=0,
)

"""
Note: this will remove all comments inside the function body also
"""


def create_dpo_dataset(
        dataset: Dataset,
        language: str,
        prompt_const: str,
        target_const: str = LABEL,
        prediction_const: str = PREDICTION
) -> Dataset:
    # add all examples that don't compile as false examples
    # ps paper proved that it's ok to have a dis-balance of positive negative examples
    nonempty = 0
    all_compile = 0

    new_dataset = []
    corrupt = args.corrupt
    for datapoint in tqdm(dataset):
        prediction = datapoint[prediction_const].replace("<｜begin▁of▁sentence｜>", "")
        prediction = prediction.replace("<｜end▁of▁sentence｜>", "")

        # Note: must use this and not same as below because prediction isn't actually a function, it has a lot more prediction_body
        prediction_body = extract_function_completion(prediction, datapoint[prompt_const])

        if language == "java":
            prompt = extract_func_def_and_docstring(datapoint, language)[1]
            prediction_body += "\n}"
        else:
            prompt = datapoint[prompt_const]

        whole_func = prompt + "\n" + prediction_body
        reference_body = extract_function_body(datapoint[target_const], language)

        if not compile_function[language](whole_func) and prediction_body != "":
            new_dataset.append(
                {
                    "prompt": prompt,
                    "rejected": prediction_body,
                    "chosen": reference_body,
                }
            )
        else:
            all_compile += 1
            if corrupt > 0:
                corrupt -= 1

                if datapoint["language"] == "python":
                    corrput_prediction = corrupt_python_code(prediction_body)
                else:
                    corrput_prediction = corrupt_java_code(prediction_body)

                new_dataset.append(
                    {
                        "prompt": prompt,
                        "rejected": prompt + "\n" + corrput_prediction,
                        "chosen": reference_body,
                    }
                )

        if prediction_body != "":
            nonempty += 1
    print("nonempty: " + str(nonempty))
    print("compile: " + str(all_compile))

    return Dataset.from_list(new_dataset)


def get_base_dataset(dataset_name: str, language: str, split: str) -> Dataset:
    return load_dataset(dataset_name, split=split, name=language)


if __name__ == "__main__":
    args = parser.parse_args()
    for split in ["train", "validation"]:
        print(split)
        base_dataset = get_base_dataset(args.base_dataset_name, args.language, split)
        out_dataset = create_dpo_dataset(base_dataset, args.language, "prepared_prompt").shuffle()
        print("len: " + str(len(out_dataset)))
        prefix = ""

        if args.corrupt > 0:
            prefix = "corrupted"

        out_dataset.push_to_hub(
            f"stojchet/{prefix}-dpo-" + args.base_dataset_name.split("/")[-1],
            token=os.getenv('HF_WRITE_TOKEN'),
            revision="main",
            config_name=args.language,
            split=split
        )

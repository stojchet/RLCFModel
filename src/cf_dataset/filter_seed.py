import argparse
import os
import re
from typing import List

from datasets import load_dataset, Dataset
from dotenv import load_dotenv

from src.cf_dataset.code_string_util import extract_func_def_and_docstring, extract_function_body
from src.cf_dataset.compiler import compile_function
from src.cf_dataset.util import get_whole_reference_function, DOCUMENTATION, CODE_COMPLETION, LANGUAGE, FUNCTION_DEFINITION

"""
This script filters the seed dataset and unifies a schema

Filter by size upper and lower limit of both documentation and code
Filter out documentation or code that contains "todo"
Add at least some part of the dataset to contain "proper" documentation

code_search_net which is the base dataset for java and python has different 
content in whole_func_string
Java: Just code
Python: Code and documentation inside the function

Parameters:
--dataset_name can be `code_search_net` or `JetBrains/KExercises` for now
--languages can be java, python, or kotlin for now
"""

parser = argparse.ArgumentParser(description="This script collects a base dataset with specified models' predictions.")
parser.add_argument(
    "--dataset_name",
    type=str,
    default="code_search_net"
)
parser.add_argument(
    "--languages",
    type=str,
    nargs="+",
    required=True,
)

args, unknown = parser.parse_known_args()


def filter_doc_by_size(datapoint, min_size: int = 60, max_size: int = 2000):
    return min_size < len(datapoint[DOCUMENTATION]) < max_size


def filter_by_spaces_count(datapoint, min_spaces: int = 5):
    return datapoint[DOCUMENTATION].count(' ') > min_spaces


# TODO: maybe remove code with todo?
def remove_docs_with_todo(datapoint):
    doc = datapoint[DOCUMENTATION].lower()
    return "todo" not in doc and "fixme" not in doc and "tbd" not in doc


def filter_code(datapoint):
    code = get_whole_reference_function(datapoint)
    methods = re.findall(r'(\w+\s+\w+\s*)\((.*?)\)', code)
    comments = len(re.findall('//.*|/\*[\s\S]*?\*/', datapoint[DOCUMENTATION]))
    lines_of_code = code.split('\n')

    return len(lines_of_code) < 70 \
           and len(re.findall(r'(if|else|for|while|case|catch)(?=\s*\(|\{)', code)) < 6 \
           and max(len(re.findall(',', params)) for _, params in methods) if methods else 0 < 8 \
                                                                                          and 1 > comments / len(
        lines_of_code) > 0.1 \
                                                                                          and (len(lines_of_code) - len(
        set(lines_of_code)) == 0)


def get_function_name_from_fdef_kotlin(prompt):
    methods = re.findall(r'(\w+\s+\w+\s*)\((.*?)\)', prompt)
    if methods == []: return ""
    return methods[0][0].replace("fun ", "")


def code_compiles(datapoint):
    return compile_function[datapoint[LANGUAGE]](get_whole_reference_function(datapoint))


def remove_method_name_mentions_in_documentation(datapoint):
    return datapoint["func_name"].split(".")[-1] not in datapoint[DOCUMENTATION].lower()


def filter_code_by_size(datapoint, min_size: int = 60, max_size: int = 4000):
    return min_size < len(datapoint[CODE_COMPLETION]) + len(datapoint[FUNCTION_DEFINITION]) < max_size


"""
Python specific filters
"""


def filter_doc_string(datapoint):
    return "\"\"\"\"" not in datapoint[DOCUMENTATION] and "\"\"\"\" " not in get_whole_reference_function(datapoint)


def filter_python_datapoint(datapoint) -> bool:
    doc = datapoint[DOCUMENTATION].lower()
    return (("return" in doc and ("args" in doc or "arguments" in doc or "param" in doc))
            or ("example::" in doc or ":param" in doc))


"""
Java specific filters
"""


def filter_java_documentation(datapoint) -> bool:
    doc = datapoint[DOCUMENTATION].lower()
    return "@param" in doc or "@return"


def all_filters(datapoint):
    if datapoint["language"] == "java":
        lang_spec = filter_java_documentation(datapoint) and remove_method_name_mentions_in_documentation(datapoint)
    elif datapoint["language"] == "python":
        lang_spec = filter_python_datapoint(datapoint) and filter_doc_string(
            datapoint) and remove_method_name_mentions_in_documentation(datapoint)
    elif datapoint["language"] == "kotlin":
        lang_spec = True
    else:
        raise ValueError("Language not supported")

    return (
            filter_code_by_size(datapoint)
            and filter_doc_by_size(datapoint)
            and filter_by_spaces_count(datapoint)
            and remove_docs_with_todo(datapoint)
            and filter_code(datapoint)
            and code_compiles(datapoint)
            and lang_spec
    )


def get_not_intersection_df(df1, df2):
    # get the indices that are in df1 but not in df2
    exclusive_indices = df1.index.difference(df2.index)

    # get the dataframe that contains the exclusive records of df1
    not_intersection_df = df1.loc[exclusive_indices]
    return not_intersection_df


def filter_dataset(dataset: Dataset) -> Dataset:
    new_dataset = dataset.filter(all_filters)

    # TODO: do i need this?
    # get_not_intersection_df(dataset.to_pandas(), new_dataset.to_pandas())[DOCUMENTATION].to_csv(
    #     f"inv-{dataset[0]['language']}.csv")

    return new_dataset


def split_datasets(dataset: Dataset, languages: List[str]) \
        -> List[Dataset]:
    return [dataset.filter(lambda x: x["language"] == lang) for lang in languages]


"""
Standardise the dataset this pipeline will work with.
After processing all datapoints will follow the following schema:
{
language: str,
documentation: str,
function_def: str,
code_completion: str,
prompt: str, (prompt will be function definition plus documentation)
}
"""


def preprocess_csn(dataset: Dataset) -> List[Dataset]:
    def modify_data(datapoint):
        function_definition, documentation, prompt = extract_func_def_and_docstring(datapoint, datapoint["language"])
        return {
            "language": datapoint["language"],
            "documentation": documentation,
            "function_def": function_definition,
            "code_completion": extract_function_body(datapoint["func_code_string"], datapoint["language"]),
            "prompt": prompt,
            "func_name": datapoint["func_name"],
        }

    datasets = split_datasets(dataset, args.languages)
    new_datasets = []
    for dataset in datasets:
        new_dataset = dataset.map(modify_data)
        new_dataset = new_dataset.remove_columns(dataset.column_names)
        new_datasets.append(new_dataset)

    return new_datasets


def preprocess_kexercises(dataset: Dataset) -> List[Dataset]:
    def modify_data(datapoint):
        function_definition, documentation, _ = extract_func_def_and_docstring(datapoint, "kotlin")
        return {
            "language": args.languages[0],
            "documentation": documentation,
            "function_def": function_definition,
            "code_completion": datapoint["solution"],
            "prompt": datapoint["problem"],
            "func_name": get_function_name_from_fdef_kotlin(datapoint["problem"]),
            "whole_func_string": datapoint["problem"] + "\n" + datapoint["solution"]
        }

    new_dataset = dataset.map(modify_data)
    new_dataset = new_dataset.remove_columns(dataset.column_names)
    new_dataset = new_dataset.filter(lambda datapoint: datapoint["function_def"] is not None)
    return [new_dataset]


if __name__ == '__main__':
    load_dotenv()

    for split in ["train", "validation", "test"]:
        og_dataset = load_dataset(args.dataset_name, split=split, trust_remote_code=True)
        # og_dataset = get_small_dataset(og_dataset.to_iterable_dataset(), 100)

        if args.dataset_name == "code_search_net":
            lang_datasets = preprocess_csn(og_dataset)
        elif args.dataset_name == "JetBrains/KExercises":
            lang_datasets = preprocess_kexercises(og_dataset)
        else:
            raise ValueError("Dataset is not supported. Please choose from 'code_search_net' or 'JetBrains/KExercises")

        for dataset, lang in zip(lang_datasets, args.languages):
            print(lang)
            print(dataset.shape)
            # act as global params for filter_all
            filtered = filter_dataset(dataset).shuffle()
            print(filtered.shape)

            filtered.push_to_hub(
                "stojchet/base_prediction_dataset",
                token=os.getenv('HF_WRITE_TOKEN'),
                revision="main",
                config_name=lang,
                split=split,
            )

# todo: run this script on both datasets and langauges to get an
#  updated dataset because of the schema change -> save in a new dataset

# todo: error when running for python
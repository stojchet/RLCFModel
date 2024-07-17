import os
import re
from typing import List

from datasets import load_dataset, Dataset
from dotenv import load_dotenv

from src.cf_dataset.compiler import compile_function

"""
Filter by size upper and lower limit of both documentation and code
Filter out documentation or code that contains "todo"
Add at least some part of the dataset to contain "proper" documentation
"""


def filter_doc_by_size(datapoint, min_size: int = 60, max_size: int = 2000):
    return min_size < len(datapoint["func_documentation_string"]) < max_size


def filter_by_spaces_count(datapoint, min_spaces: int = 5):
    return datapoint["func_documentation_string"].count(' ') > min_spaces


def remove_docs_with_todo(datapoint):
    doc = datapoint["func_documentation_string"].lower()
    return "todo" not in doc and "fixme" not in doc and "tbd" not in doc


def remove_method_name_mentions_in_documentation(datapoint):
    return datapoint["func_name"].split(".")[-1] not in datapoint["func_documentation_string"].lower()


def filter_code(datapoint):
    if datapoint["language"] == "java":
        code = datapoint["whole_func_string"]
    else:
        code = datapoint["whole_func_string"].replace(datapoint["func_documentation_string"], "")

    methods = re.findall(r'(\w+\s+\w+\s*)\((.*?)\)', code)
    comments = len(re.findall('//.*|/\*[\s\S]*?\*/', datapoint["func_documentation_string"]))
    lines_of_code = code.split('\n')

    return len(lines_of_code) < 70 \
        and len(re.findall(r'(if|else|for|while|case|catch)(?=\s*\(|\{)', code)) < 6 \
        and max(len(re.findall(',', params)) for _, params in methods) if methods else 0 < 8 \
        and 1 > comments / len(lines_of_code) > 0.1 \
        and (len(lines_of_code) - len(set(lines_of_code)) == 0)


def code_compiles(row):
    return compile_function[row["language"]](row["whole_func_string"])


"""
Python specific filters
"""


def filter_code_by_size_python(datapoint, min_size: int = 60, max_size: int = 4000):
    return min_size < len(datapoint["whole_func_string"]) - len(datapoint["func_documentation_string"]) < max_size


def filter_doc_string(datapoint):
    return "\"\"\"\"" not in datapoint["whole_func_string"]


def filter_python_datapoint(datapoint) -> bool:
    doc = datapoint["func_documentation_string"].lower()
    return (("return" in doc and ("args" in doc or "arguments" in doc or "param" in doc))
            or ("example::" in doc or ":param" in doc))


"""
Java specific filters
"""


def filter_code_by_size_java(datapoint, min_size: int = 60, max_size: int = 4000):
    return min_size < len(datapoint["whole_func_string"]) < max_size


def filter_java_documentation(datapoint) -> bool:
    doc = datapoint["func_documentation_string"].lower()
    return "@param" in doc or "@return"


def all_filters(datapoint):
    if datapoint["language"] == "java":
        lang_spec = filter_java_documentation(datapoint) and filter_code_by_size_java(datapoint)
    else:
        lang_spec = filter_python_datapoint(datapoint) and filter_code_by_size_python(datapoint) and filter_doc_string(datapoint)

    return (
        filter_doc_by_size(datapoint)
        and filter_by_spaces_count(datapoint)
        and remove_docs_with_todo(datapoint)
        and remove_method_name_mentions_in_documentation(datapoint)
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

    get_not_intersection_df(dataset.to_pandas(), new_dataset.to_pandas())["func_documentation_string"].to_csv(f"inv-{dataset[0]['language']}.csv")

    return new_dataset


def split_datasets(dataset: Dataset, languages: List[str]) \
        -> List[Dataset]:
    return [dataset.filter(lambda x: x["language"] == lang) for lang in languages]


if __name__ == '__main__':
    load_dotenv()
    dataset_name: str = 'code_search_net'

    for split in ["train", "validation", "test"]:
        og_dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)

        languages = ["java", "python"]
        lang_datasets = split_datasets(og_dataset, languages)

        for dataset, lang in zip(lang_datasets, languages):
            print(lang)
            print(dataset.shape)
            filtered = filter_dataset(dataset).shuffle()
            print(filtered.shape)

            filtered.push_to_hub(
                "stojchet/csn_java_python_subset",
                token=os.getenv('HF_WRITE_TOKEN'),
                revision="main",
                config_name=lang,
                split=split,
            )

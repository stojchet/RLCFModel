from functools import partial

import torch
from datasets import Dataset, IterableDataset


def get_small_dataset(dataset: IterableDataset, n: int = 100) -> Dataset:
    dataset = dataset.take(n)

    def gen_from_iterable_dataset(iterable_ds):
        yield from iterable_ds

    dataset = Dataset.from_generator(partial(gen_from_iterable_dataset, dataset), features=dataset.features)
    return dataset


def dtype_from_string(dtype_str):
    dtype = getattr(torch, dtype_str)
    return dtype

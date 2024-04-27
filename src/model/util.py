from functools import partial

from datasets import Dataset


def get_small_dataset(dataset: Dataset, n: int = 100) -> Dataset:
    dataset = dataset.take(n)

    def gen_from_iterable_dataset(iterable_ds):
        yield from iterable_ds

    dataset = Dataset.from_generator(partial(gen_from_iterable_dataset, dataset), features=dataset.features)
    return dataset

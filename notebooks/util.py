import collections
import matplotlib.pyplot as plt


def bucketize(lengths, bucket_size):
    bucket_dict = collections.defaultdict(int)

    for length in lengths:
        if length > 2000:
            length = 1000

        bucket = length // bucket_size
        bucket_dict[bucket] += 1

    print(bucket_dict)
    selected_buckets = [(f'{bucket * bucket_size}-{(bucket + 1) * bucket_size - 1}', count) for bucket, count in
                        bucket_dict.items() if count > 0]
    selected_buckets = sorted(selected_buckets, key=lambda x: int(x[0].split("-")[0]))
    selected_buckets[-1] = (selected_buckets[-1][0].split("-")[0] + "-" + str(max(lengths)), selected_buckets[-1][1])
    x_values, y_values = zip(*selected_buckets)

    plt.bar(x_values, y_values, color='blue')

    plt.xlabel('Bucket (Length range)')
    plt.ylabel('Frequency')
    plt.title('Frequency Distribution of Reference Lengths in Characters')
    plt.xticks(rotation=45)

    plt.show()
    return selected_buckets


# from datasets import load_dataset, concatenate_datasets
#
# final = load_dataset(f"stojchet/base_py_java", split="train", name="java", trust_remote_code=True)
#
# df = final.to_pandas()
# ...res = df["whole_func_string"].value_counts()
#
# print(...res[...res > 1])


import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset




ref = load_dataset(f"stojchet/base_py_java", split="train", name="python", trust_remote_code=True).to_pandas()

def get_non_intersection(df):
    s1 = pd.Series(list(set(df["whole_func_string"])))
    s2 = pd.Series(list(set(ref["whole_func_string"])))
    bool_series = s1.isin(s2)

    non_intersect_indices = np.where(~bool_series)[0]
    # print(non_intersect_indices)
    non_intersect_values = df.iloc[non_intersect_indices]
    # print(non_intersect_values)

    a = pd.Series(list(set(df["whole_func_string"]).intersection(set(ref["whole_func_string"]))))
    print(len(pd.Series(list(set(a).intersection(set(non_intersect_values["whole_func_string"]))))))
    print(len(pd.Series(
        list(set(ref["whole_func_string"]).intersection(set(non_intersect_values["whole_func_string"]))))))

    return non_intersect_values



rest = [
    Dataset.from_pandas(get_non_intersection(load_dataset(f"stojchet/ds_2-empty", split="train", name="python", revision="4.0", trust_remote_code=True).to_pandas())),
    Dataset.from_pandas(get_non_intersection(load_dataset(f"stojchet/ds_3-empty", split="train", name="python", revision="4.0", trust_remote_code=True).to_pandas())),
    Dataset.from_pandas(get_non_intersection(load_dataset(f"stojchet/ds_4-empty", split="train", name="python", revision="4.0", trust_remote_code=True).to_pandas()))
]
rest
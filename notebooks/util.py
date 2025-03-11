import collections
import matplotlib.pyplot as plt


def tokenize_and_bucketize(tokenizer, dataset, dataset_key, bucket_size=100):
    tokenized_sizes = []

    for datapoint in dataset:
        tokenized_sizes.append(len(tokenizer.tokenize(datapoint[dataset_key])))
    bucketize(tokenized_sizes, bucket_size)

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

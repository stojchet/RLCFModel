import re

import numpy as np
from datasets import Dataset
from tqdm import tqdm

from src.evaluate.sanitize.function_extraction import extract_function_completion


def sanitize_dataset(dataset: Dataset, base_prompt: str,
                     predictions_field: str= "solution", prompt_field: str = "prompt"):
    """
    This function sanitizes a dataset by extracting function completions
    from the model output for each data point in the dataset.
    """
    sanitized_predictions = []
    prep_text = []
    language = dataset[0]["language"]

    for i, line in tqdm(enumerate(dataset)):
        prediction = line[predictions_field]

        code = extract_function_completion(prediction, line[prompt_field])

        if language == "java":
            code += "\n}"

        prep_text.append(prediction)
        sanitized_predictions.append(code)

    dataset = dataset.rename_column(predictions_field, "original_prediction")
    dataset = dataset.add_column("middle_prediction", prep_text)

    sanitized_predictions = np.array(sanitized_predictions, dtype=str)
    dataset = dataset.add_column("completion", sanitized_predictions)

    return dataset

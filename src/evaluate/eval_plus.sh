#!/bin/bash

predictions_file_name="$1"
model_name="$2"
path_to_prompt="$3"
language="$4"
dataset="mbpp"


python3 src/evaluate/evalp_collect_predictions.py --predictions_file_name="$predictions_file_name" --model_name="$model_name" --path_to_prompt="$path_to_prompt" --language="$language"
python3 evalplus.sanitize --samples predictions_file_name
python3 evalplus.evaluate --samples predictions_file_name --dataset $dataset
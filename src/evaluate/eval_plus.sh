#!/bin/bash

predictions_file_name="$1"
model_name="$2"
path_to_prompt="$3"
language="$4"

case $language in
    "python")
        dataset="mbpp"
        ;;
    "java")
        dataset="mbjp"
        ;;
    *)
        echo "Unknown language"
        exit 1
        ;;
esac


python3 src/evaluate/evalp_collect_predictions.py --$predictions_file_name="$predictions_file_name" --model_name="$model_name" --path_to_prompt="$path_to_prompt" --language="$language"
python3 tools/sanitize.py --samples predictions_file_name --dataset $dataset
python3 evalplus.evaluate --samples predictions_file_name --dataset $dataset
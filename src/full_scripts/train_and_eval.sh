#!/bin/bash

export PYTHONPATH=.

config_path="$1"
config_name="$2"
path_to_prompt="$3"
batch_size="$4"


echo "TRAINING MODEL"
python3 src/model/sft_model.py --config_path="$config_path" --config_name="$config_name"
echo "COLLECTING PREDICTIONS"
python3 src/evaluate/collect_predictions.py --config_path="$config_path" --config_name="$config_name" --path_to_prompt="$path_to_prompt" --batch_size="$batch_size"
echo "RUNNING EVALUATION"
python3 src/evaluate/run_eval.py --config_path="$config_path" --config_name="$config_name" --prompt_path="$path_to_prompt"

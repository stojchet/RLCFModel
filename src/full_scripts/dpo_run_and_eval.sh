#!/bin/bash

export PYTHONPATH=.

config_name="$1"
config_path="$2"
sft_config_name="$3"
sft_config_path="$4"
path_to_prompt="$5"
batch_size="$6"

echo "DPO MODEL"
python3 src/model/dpo_model.py --config_path="$config_path" --config_name="$config_name"
echo "SFT MODEL"
python3 src/model/sft_model.py --config_path="$sft_config_path" --config_name="$sft_config_name" --add_config_path="$config_path" --add_config_name="$config_name"
echo "COLLECTING PREDICTIONS DPO"
python3 src/evaluate/collect_predictions.py --config_path="$config_path" --config_name="$config_name"
echo "COLLECTING PREDICTIONS"
python3 src/evaluate/collect_predictions.py --config_path="$sft_config_path" --config_name="$sft_config_name" --add_config_path="$config_path" --add_config_name="$config_name" --path_to_prompt="$path_to_prompt" --batch_size="$batch_size"
echo "RUNNING EVALUATION"
python3 src/evaluate/run_eval.py --config_path="$config_path" --config_name="$config_name" --prompt_path="$path_to_prompt"

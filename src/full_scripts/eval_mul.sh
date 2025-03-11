#!/bin/bash

export PYTHONPATH=.

path_to_prompt="src/prompts/python/empty.txt"


DIR_PATH="configs/big"
# shellcheck disable=SC2045
for file in $(ls $DIR_PATH)
do
  filename=$(basename "$file" .yaml)
  echo $filename
  python3 src/evaluate/run_eval.py --add_config_path=$DIR_PATH --add_config_name="$filename" --prompt_path="$path_to_prompt" --config_path="configs/no_lora" --config_name="sft8"
done

python-d3-empty

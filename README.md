# Reinforcement Learning using Compiler Feedback Model
The goal of this project is to course tune a pre-trained LLM using compiler generated RL feedback on the task of generating code based on textual explanation. 

The goal of the coarse tuning is to align the model's output with compilable code more often. The way to do it is by formulating the problem of writing code as an MDP. The policy in this definition is the pre-trained model and we train a reward model consisting of two parts: a compiler and a discriminator. 

### Requirements
* Python >3.9
* To install all requirements run: `pip install requirements.txt`

#### Tokens
Create a .env file and add your `HF_WRITE_TOKEN`.

## Train model

### Run SFT
In order to train an sft model, first create a yaml config specifying the hyperparameters.

```yaml
max_seq_length: 1500
language: python
dataset_size: 10000
epochs: 1
per_device_train_batch_size: 8
gradient_accumulation_steps: 16
learning_rate: 1.41e-6
lora_r: 64
lora_alpha: 16
lora_dropout: 0.05
dataset_name: stojchet/csn_java_python_subset
base_model: deepseek-ai/deepseek-coder-1.3b-base
dataset_ref_field: whole_func_string
```

Run the script `src/model/sft_model.py` with the above created config. 

```shell
python3 src/model/sft_model.py --config_path configs --config_name sft
```

### Collect base dataset

Create a yaml config for the hyperparameters
```yaml
dataset_name: "stojchet/csn_java_python_subset"
batch_size: 1
model_name: "deepseek-ai/deepseek-coder-1.3b-base"
max_new_tokens: 1000
torch_dtype: "bfloat16"
dataset_size: "inf"
```

Run the script for collecting base dataset
```shell
python3 base_dataset.py \
--language=java \
--prompt_path=src/prompts/python/empty.txt \
--config_path configs/base
--config_name deepseek_bs1
```

The final dataset will be saved as {config_name}-{prompt_name}, in above case deepseek_bs1-empty

### Run DPO
#### Collect Dataset
1. Collect base dataset by running `src/cf_dataset/base_dataset.py`
2. Collect final preference enriched dataset `src/cf_dataset/dpo_dataset.py`

```shell
python3 dpo_dataset.py \
--base_dataset_name stojchet/deepseek_bs1-empty \
--language python 
```

The output dataset will be stored as stojchet/dpo-deepseek_bs1-empty

#### Train model
1. Train the model with DPO by running `src/model/dpo_model.py`
    
Create a DPO hyperparameter config with the dpo dataset specified
```yaml
max_seq_length: 1000
language: python
dataset_size: 5000
epochs: 1
per_device_train_batch_size: 2
gradient_accumulation_steps: 64
learning_rate: 0.001
no_lora: True
lora_r: 64
lora_alpha: 16
lora_dropout: 0.05
weight_decay: 0.1
dataset_name: stojchet/dpo-deepseek_bs1-empty
base_model: "deepseek-ai/deepseek-coder-1.3b-base"
```

Train the DPO model
```shell
python3 src/model/dpo_model.py --config_path=dpo_conf_path --config_name=dpo_conf
```

Train an SFT model on top of the DPO model
```shell
python3 src/model/sft_model.py --config_path=sft_conf_path --config_name=sft_conf --add_config_path=dpo_conf_path --add_config_name=dpo_conf
```

### Run KTO
#### Collect Dataset
1. Collect base dataset by running `src/cf_dataset/base_dataset.py`
2. Create KTO dataset by running `src/cf_dataset/kto_dataset.py`

```shell
python3 kto_dataset.py \
--base_dataset_name stojchet/deepseek_bs1-empty \
--language python 
```

The output dataset will be stored as stojchet/kto-deepseek_bs1-empty


#### Train model
1. Train the model with KTO by running `src/model/kto_model.py`

Create a KTO hyperparameter config with the kto dataset specified
```yaml
max_seq_length: 1000
language: python
dataset_size: 5000
epochs: 1
per_device_train_batch_size: 8
gradient_accumulation_steps: 16
learning_rate: 1e-5
no_lora: True
lora_r: 64
lora_alpha: 16
lora_dropout: 0.05
warmup_ratio: 0.1
weight_decay: 0.1
dataset_name: stojchet/kto-deepseek_bs1-empty
base_model: deepseek-ai/deepseek-coder-1.3b-base
```

Train the KTO model
```shell
python3 src/model/kto_model.py --config_path=kto_conf_path --config_name=kto_conf
```

Train an SFT model on top of the DPO model
```shell
python3 src/model/sft_model.py --config_path=sft_conf_path --config_name=sft_conf --add_config_path=kto_conf_path --add_config_name=kto_conf
```

## Evaluate
Run `collect_predictions.py` for the specific model.
To evaluate the plain KTO/DPO model
```shell
python3 src/evaluate/collect_predictions.py --config_path=kto_path --config_name=kto_conf
```

To evaluate the KTO/DPO model which is then SFT trained
```shell
python3 src/evaluate/collect_predictions.py --config_path=sft_conf_path --config_name=sft_conf --add_config_path=kto_conf_path --add_config_name=kto_conf --path_to_prompt="$path_to_prompt" --batch_size="$batch_size"
```

To get final metrics and upload them to wandb
```shell
python3 src/evaluate/run_eval.py --config_path=sft_conf_path --config_name=sft_conf --add_config_path=kto_conf_path --add_config_name=kto_conf --path_to_prompt="$path_to_prompt" --batch_size="$batch_size"
```


## Scripts

1. Train nd evaluate sft
```shell
./src/full_scripts/train_and_eval.sh configs sft_conf src/prompt/python/empty.txt 1
```
2. Train and evaluate KTO
```shell
./src/full_scripts/kto_run_and_eval.sh configs/ktos kto_conf configs sft_conf src/prompt/python/empty.txt 1
```

3. Train and evaluate DPO
 ```shell
./src/full_scripts/dpo_run_and_eval.sh configs/dpos dpo_conf configs sft_conf src/prompt/python/empty.txt 1
```

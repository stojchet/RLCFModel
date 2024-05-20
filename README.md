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
Run the script `src/model/sft_model.py`. 

### Run DPO
#### Collect Dataset
1. Collect base dataset by running `src/cf_dataset/base_dataset.py`
2. Train discriminator using hte above dataset by running `src/cf_dataset/discriminator.py`
3. Collect final preference enriched dataset `src/cf_dataset/disc_preference_dataset.py`
4. Train the model with DPO by running `src/model/dpo_model.py`
    
## Evaluate pre-trained model
Run `src/evaluate/eval_plus.sh`. Evaluates model on MBPP or HumanEval using EvalPlus
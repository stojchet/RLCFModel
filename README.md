# Reinforcement Learning using Compiler Feedback Model
The goal of this project is to course tune a pre-trained LLM using compiler generated RL feedback on the task of generating code based on textual explanation. 

The goal of the coarse tuning is to align the model's output with compilable code more often. The way to do it is by formulating the problem of writing code as an MDP. The policy in this definition is the pre-trained model and we train a reward model consisting of two parts: a compiler and a discriminator. 

### Requirements
* Python >3.9
* To install all requirements run: `pip install requirements.txt`
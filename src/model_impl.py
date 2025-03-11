import gc
from typing import Union, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


"""
This class wraps a transformer model with methods for generating predictions from prompts.
It uses the HuggingFace AutoModelForCausalLM and AutoTokenizer classes to perform tokenization and generation.
"""

class Model:
    def __init__(self,
                 name: str,
                 torch_dtype: torch.dtype = torch.float32,
                 max_new_tokens: int = 1000,
                 padding: Union[bool, str] = False,
                 truncation: Union[bool, str] = False) -> None:
        print(torch_dtype)
        self.name = name
        self.max_new_tokens = max_new_tokens
        self.padding = padding
        self.truncation = truncation

        self.model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype=torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True, use_fast=True)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id

    def predict(self, prompts: List[str], attention_mask=None) -> str:
        """
        This method generates a list of predictions from a list of string prompts.
        It tokenizes the prompts, generates outputs from the model, and decodes the outputs into a list of strings.

        Parameters:
        prompts (List[str]): The list of prompts for the model to generate from.

        Returns:
        List[str]: The list of generated strings.

        """
        # attention mask
        tokenized = self.tokenizer(prompts,
                                   return_tensors="pt",
                                   truncation=self.truncation,
                                   padding=True)

        outputs = self.model.generate(**tokenized,
                                      max_new_tokens=self.max_new_tokens,
                                      do_sample=True,
                                      pad_token_id=self.tokenizer.pad_token_id,
                                      )
        result = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        torch.cuda.empty_cache()
        gc.collect()

        return result

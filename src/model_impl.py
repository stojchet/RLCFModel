import gc
from typing import Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Model:
    def __init__(self,
                 name: str,
                 torch_dtype: torch.dtype = torch.float32,
                 max_new_tokens: int = 2000,
                 padding: Union[bool, str] = False,
                 truncation: Union[bool, str] = False) -> None:
        print(torch_dtype)
        self.name = name
        self.max_new_tokens = max_new_tokens
        self.padding = padding
        self.truncation = truncation
        self.model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype=torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True, use_fast=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def predict(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=False, truncation=self.truncation, padding=self.padding)

        outputs = self.model.generate(**inputs,
                                      max_new_tokens=self.max_new_tokens,
                                      do_sample=True,
                                      pad_token_id=self.tokenizer.pad_token_id
                                      )
        result = self.tokenizer.batch_decode(outputs)[0]

        torch.cuda.empty_cache()
        gc.collect()

        return result

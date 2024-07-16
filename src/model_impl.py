import gc
from typing import Union, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration


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

        if self.__is_code_t5():
            self.model = T5ForConditionalGeneration.from_pretrained(name)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype=torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True, use_fast=True)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id

    def predict_single(self, prompt: str) -> str:
        if self.__is_code_t5():
            prompt = prompt + "<extra_id_0>"

        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=False, truncation=self.truncation,
                                padding=self.padding)

        outputs = self.model.generate(**inputs,
                                      max_new_tokens=self.max_new_tokens,
                                      do_sample=True,
                                      pad_token_id=self.tokenizer.pad_token_id
                                      )
        result = self.tokenizer.batch_decode(outputs)[0]

        torch.cuda.empty_cache()
        gc.collect()

        return result

    def predict(self, prompts: List[str]) -> str:
        if self.__is_code_t5():
            prompts = [prompt + "<extra_id_0>" for prompt in prompts]

        # attention mask
        tokenized = self.tokenizer(prompts, return_tensors="pt", return_attention_mask=False,
                                   truncation=self.truncation, padding=True)

        outputs = self.model.generate(**tokenized,
                                      max_new_tokens=self.max_new_tokens,
                                      do_sample=True,
                                      pad_token_id=self.tokenizer.pad_token_id,
                                      )
        result = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        torch.cuda.empty_cache()
        gc.collect()

        return result

    def __is_code_t5(self):
        return "codet5" in self.name.lower()

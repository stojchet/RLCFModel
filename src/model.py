import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Model:
    def __init__(self,
                 name: str,
                 torch_dtype: torch.dtype,
                 max_new_tokens: int = 2000) -> None:
        self.name = name
        self.max_new_tokens = max_new_tokens
        self.model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype=torch_dtype).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True, use_fast=True)

    def predict(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=False)

        outputs = self.model.generate(**inputs,
                                      max_new_tokens=self.max_new_tokens,
                                      do_sample=True,
                                      )
        return self.tokenizer.batch_decode(outputs)[0][len(prompt):]

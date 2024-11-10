import torch
from .template import TEMPLATE


def rag_prompt_template(tokenizer):
    output_tensor = tokenizer.apply_chat_template(
        TEMPLATE, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    return output_tensor
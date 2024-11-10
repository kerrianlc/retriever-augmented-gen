from transformers import pipeline
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from .common import READER_MODEL_NAME


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def reader_llm():
    # Insighful explanation of why to choose AutoModelForCausalLM instead of AutoModel
    # https://www.reddit.com/r/huggingface/comments/1bv1kfk/what_is_the_difference_between/?rdt=39865
    # Studying the embedded representation could be useful as well so the unused import
    model = AutoModelForCausalLM.from_pretrained(
        READER_MODEL_NAME, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
    if torch.cuda.is_available():
        model = model.to("cuda")
    else:
        print("CUDA is not available. Running on CPU instead.")
    params = {
        "model": model,
        "tokenizer": tokenizer,
        "task": "text-generation",
        "do_sample": True,
        "temperature": 0.2,
        "repetition_penalty": 1.1,
        "return_full_text": False,
        "max_new_tokens": 500,
    }
    return pipeline(**params), tokenizer

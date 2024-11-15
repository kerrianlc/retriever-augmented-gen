import pandas as pd
import numpy as np
import random
from tqdm.notebook import trange
from collections import defaultdict
import torch.nn.functional as F

def tokenSHAP(prompt, model, tokenizer, ratio=1 / 16):
    """
    Computes SHAP values for individual tokens in a text prompt based on their 
    contribution to the output embeddings of a given model.

    Args:
        prompt (str): Input text prompt.
        model (callable): A function/model that processes the prompt and tokenizer 
                          and returns a dictionary with "hidden embedding".
        tokenizer (callable): Tokenizer compatible with the model.
        ratio (float): Fraction of possible token combinations to sample for SHAP values.

    Returns:
        pd.Series: Normalized SHAP values for each token in the prompt.
    """
    def random_combination(tokens):
        """
        Randomly selects a subset of tokens.
        """
        n = len(tokens)
        subset_size = random.randint(1, n - 2)  # Exclude empty and full set
        return random.sample(tokens, subset_size)

    # Tokenize the prompt
    words = prompt.split()
    if not words:
        raise ValueError("The prompt cannot be empty.")
    
    # Baseline embedding for the full prompt
    baseline_output = model(prompt, tokenizer)
    baseline_embedding = baseline_output["hidden embedding"]

    # Dictionaries to store cosine similarities
    without_contribs = defaultdict(list)
    with_contribs = defaultdict(list)
    num_words = len(words)
    
    # Calculate number of random samples
    total_combinations = 2 ** num_words - 1  
    sample_size = int(total_combinations * ratio) - num_words

    # Compute contributions when one token is removed
    for i in range(num_words):
        modified_prompt = " ".join(words[:i] + words[i + 1:])
        modified_embedding = model(modified_prompt, tokenizer)["hidden embedding"]
        
        cs = F.cosine_similarity(modified_embedding, baseline_embedding).item()
        without_contribs[words[i]].append(cs)
        
        # Distribute contributions to 'with_contribs' for remaining words
        for j, word in enumerate(words):
            if j != i:
                with_contribs[word].append(cs)

    # Random combinations of tokens
    for _ in trange(sample_size, desc="Sampling SHAP contributions"):
        selected_tokens = random_combination(words)
        missing_tokens = set(words) - set(selected_tokens)
        
        selected_prompt = " ".join(selected_tokens)
        selected_embedding = model(selected_prompt, tokenizer)["hidden embedding"]
        
        cs = F.cosine_similarity(selected_embedding, baseline_embedding).item()
        
        for word in missing_tokens:
            without_contribs[word].append(cs)
        for word in selected_tokens:
            with_contribs[word].append(cs)

    shap_df = pd.DataFrame({
        "token": words,
        "withouts": [np.mean(without_contribs[word]) for word in words],
        "withs": [np.mean(with_contribs[word]) for word in words],
    })
    shap_df["shap"] = shap_df["withs"] - shap_df["withouts"]
    
    # Normalize SHAP values
    shap_df["shap"] /= np.sqrt(np.sum(shap_df["shap"] ** 2))
    
    return shap_df.set_index("token")["shap"]
import tiktoken
import torch


def produce_input_embeddings(
    prompts: list[str], padding_value: int = 0
) -> torch.Tensor:
    tokenizer = tiktoken.get_encoding("gpt2")
    embeddings = [tokenizer.encode(prompt) for prompt in prompts]
    max_len = max(len(embedding) for embedding in embeddings)

    padded = [
        embedding + [padding_value] * (max_len - len(embedding))
        for embedding in embeddings
    ]
    return torch.stack([torch.tensor(embedding) for embedding in padded], dim=0)

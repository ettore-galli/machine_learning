# https://medium.com/@lahsaini/building-an-llm-from-scratch-the-foundational-layer-and-structural-stability-fdc1fd668dc8

import torch
from torch.utils.data import Dataset, DataLoader


def show_tensor(label: str, dsptensor: torch.Tensor):
    print(f"\n ----- {label} ----- \n")
    print("\ndata:\n")
    print(dsptensor)
    print("\shape:\n")
    print(dsptensor.shape)


class CustomLayerNorm(torch.nn.Module):
    def __init__(self, embedding_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # Trainable scale adjustments initialized to 1
        self.gamma = torch.nn.Parameter(torch.ones(embedding_dim))
        # Trainable translation parameters initialized to 0
        self.beta = torch.nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input state shapes: (B, T, C)
        # CRITICAL: keepdim=True preserves the 3D grid layout, avoiding channel collapse
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Center and scale standard outputs
        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        # Apply learnable parameter deformations
        return self.gamma * x_hat + self.beta


def embedding():
    vocab_size = 65  # Number of different characters in Shakespeare's text
    embedding_dim = 8  # The dimension of each embedding vector
    # Creates a dictionary of vector embeddings for tokens
    token_embedding_table = torch.nn.Embedding(vocab_size, embedding_dim)

    input_data = [24, 43, 58, 5, 57, 1, 46, 43, 27, 11]
    input_indices = torch.tensor([input_data])

    token_embeddings = token_embedding_table(input_indices)

    show_tensor("token embeddings", token_embeddings)

    # Creates a layer that converts 8 positions into 8-dimensional embedding vectors
    position_embedding_layer = torch.nn.Embedding(len(input_data), embedding_dim)

    # To retrieve these embeddings, we need to feed the layer a sequence of position indices. Instead of passing in text IDs, we simply pass in consecutive integers counting up from zero.
    # Creates a tensor with consecutive integers from 0 to 7
    position_indices = torch.arange(len(input_data))

    # Fetch the unique spatial vectors for these 8 positions
    position_embeddings = position_embedding_layer(position_indices)

    # show_tensor("position embeddings", position_embeddings)

    final_embeddings = token_embeddings + position_embeddings

    # show_tensor("final embeddings", final_embeddings)

    norm = CustomLayerNorm(embedding_dim=embedding_dim)

    norm_token_embeddings = norm.forward(token_embeddings)

    show_tensor("norm token embeddings", norm_token_embeddings)

def broadcast():
    a = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    b = torch.Tensor([[10, 100, 1000], [20, 2000, 2000]])

    print(a + b)


if __name__ == "__main__":
    embedding()
    # broadcast()

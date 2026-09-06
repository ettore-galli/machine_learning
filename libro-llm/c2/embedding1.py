import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

def embedding():
    input_ids=torch.tensor([2,1,3,4,2,5,7,5,3])
    torch.manual_seed(112358)
    vocab_size=50257
    output_dim=256
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    print(embedding_layer.weight)


if __name__ == "__main__":
    embedding()
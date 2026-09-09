import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

example_file = "the-verdict.txt"


def show_tensor(label: str, dsptensor: torch.Tensor):
    print(f"\n ----- {label} ----- \n")
    print("\ndata:\n")
    print(dsptensor)
    print("\shape:\n")
    print(dsptensor.shape)


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + stride : i + max_length + stride]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

        print(self.input_ids)
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt, batch_size, max_length, stride, shuffle=True, drop_last=True, num_workers=0
):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


def self_attention(inputs: torch.Tensor):
    floats = inputs.float()  
    attention = torch.empty(floats.shape[0], floats.shape[0])
    for i, item_i in enumerate(floats):
        for j, item_j in enumerate(floats):
            attention[i, j] = torch.dot(item_i, item_j)
          
    return torch.softmax(attention-attention.max(), dim=0)


def embedding():
    torch.manual_seed(1)
    vocab_size = 50257
    output_dim = 3
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    # print(token_embedding_layer.weight)

    raw_text = "The quick brown fox jumps over the lazy wolf because we need a larger sentence as an example"

    max_length = 5

    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=7,
        max_length=max_length,
        stride=1,
        shuffle=False,
    )

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    # # print(inputs)

    token_embeddings = token_embedding_layer(inputs[0])

    # context_length = max_length
    # pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    # pos_embeddings = pos_embedding_layer(torch.arange(context_length))

    show_tensor("Inputs", inputs)
    show_tensor("Targets", targets)
    show_tensor("Token embeddings", token_embeddings)

    embeddings_attention = self_attention(inputs=token_embeddings)

    show_tensor("Embeddings Attention", embeddings_attention)

    print


if __name__ == "__main__":
    embedding()

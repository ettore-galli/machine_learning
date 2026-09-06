import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

example_file = "the-verdict.txt"


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


def embedding():
    input_ids = torch.tensor([2, 1, 3, 4, 2, 5, 7, 5, 3])
    torch.manual_seed(112358)
    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    # print(token_embedding_layer.weight)

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    max_length = 4

    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=8,
        max_length=max_length,
        stride=max_length - 1,
        shuffle=False,
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    # print(inputs)
    # print(inputs.shape)

    token_embeddings = token_embedding_layer(inputs)

    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))

    print("_" * 78)
    print("\nInputs:\n")
    print(inputs)
    print(inputs.shape)

    print("_" * 78)
    print("\nTargets:\n")
    print(targets)
    print(targets.shape)

    print("_" * 78)
    print("\nToken embeddings:\n")
    print(token_embeddings)
    print(token_embeddings.shape)

    print("_" * 78)
    print("\nPos embeddings:\n")
    print(pos_embeddings)
    print(pos_embeddings.shape)


if __name__ == "__main__":
    embedding()

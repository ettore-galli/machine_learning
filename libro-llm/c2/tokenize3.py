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


def use_dataloader():
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    vocab_size = 50257
    output_dim = 256
    context_length = 1024

    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

    batch_size = 8
    max_length = 4
    dataloader = create_dataloader_v1(
        raw_text, batch_size=batch_size, max_length=max_length, stride=max_length
    )
    for batch in dataloader:
        x, y = batch

        token_embeddings = token_embedding_layer(x)
        pos_embeddings = pos_embedding_layer(torch.arange(max_length))

        input_embeddings = token_embeddings + pos_embeddings

        print("-" * 78)
        print(pos_embeddings)

        print("-" * 78)
        print(input_embeddings)

        break

    # print(input_embeddings.shape)
    # print(input_embeddings)


def use_dataloader(dataloader):
    for batch in dataloader:
        x, y = batch

        token_embeddings = token_embedding_layer(x)
        pos_embeddings = pos_embedding_layer(torch.arange(max_length))

        input_embeddings = token_embeddings + pos_embeddings

        print("-" * 78)
        print(pos_embeddings)

        print("-" * 78)
        print(input_embeddings)

        break

    # print(input_embeddings.shape)
    # print(input_embeddings)


def try_dataloader():
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    vocab_size = 50257
    output_dim = 256
    context_length = 1024

    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

    batch_size = 8
    max_length = 4

    dataloader = create_dataloader_v1(
        raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
    )

    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)

    for stride in [1,2]:
        print("stride", stride)
        tokenizer = tiktoken.get_encoding("gpt2")
        dataset = GPTDatasetV1(raw_text, tokenizer, max_length, stride=stride)

        print(dataset.input_ids[0], dataset.target_ids[0])

if __name__ == "__main__":
    try_dataloader()

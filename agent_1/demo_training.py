import logging

from datasets import load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def demo():
    raw_datasets = load_dataset("glue", "mrpc")

    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
    # tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])

    def tokenize_function(batch):
        print("TOKENIZE batch", type(batch))
        return tokenizer(
            batch["sentence1"],
            batch["sentence2"],
            padding="max_length",
            truncation=True,
        )

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    _ = tokenizer.convert_ids_to_tokens(tokenized_datasets["train"][1]["input_ids"])

    print(".")


if __name__ == "__main__":
    demo()

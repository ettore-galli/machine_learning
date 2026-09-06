example_file = "the-verdict.txt"

import re


def tokenize_content(corpus: str) -> list[str]:
    splitted = re.split(r"([,.;:?!'\"(\\/)]|--|\s)", corpus)
    tokenized = [token for token in splitted if token]
    return tokenized


def tokenize_file(input_file: str) -> list[str]:
    with open(input_file, "r", encoding="utf-8") as ifile:
        corpus = ifile.read()
        return tokenize_content(corpus)


def create_embedding_dict(tokenized: list[str]):
    print("creating set...")
    tokens = set(tokenized)
    print("creating dict...")
    return {token: index for index, token in enumerate(tokens)}


class Tokenizer:
    def __init__(self, embedding_dict):
        self.embedding_dict = embedding_dict
        self.reverse_map = {value: key for key, value in embedding_dict.items()}

    def encode_text(self, text: str) -> list[int]:
        return self.encode(tokenize_content(corpus=text))

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.embedding_dict.get(token, -1) for token in tokens]

    def decode(self, encoded: list[int]) -> list[str]:
        return [self.reverse_map.get(item, "[???]") for item in encoded]


if __name__ == "__main__":
    tokenized_file = tokenize_file(example_file)

    embedding_dict = create_embedding_dict(tokenized_file)

    tokenizer = Tokenizer(embedding_dict=embedding_dict)

    sentence = "worth His of open told qwerty past"
    
    print(sentence)

    encoded = tokenizer.encode_text(sentence)
    print(encoded)

    decoded = tokenizer.decode(encoded)
    print(decoded)

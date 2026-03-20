import re
from typing import List

from tokenizer import create_word_maps, remove_tokens, tokenize_text


def split_large_file(filename: str) -> List[str]:
    with open(filename, "r", encoding="utf-8") as text_file:
        text = text_file.read()

        return remove_tokens(tokenize_text(text), [" "])


def split_demo():
    tokens = split_large_file(filename="./data/the-verdict.txt")
    print(tokens)
    word_to_id, id_to_word = create_word_maps(tokens)
    print(word_to_id)


if __name__ == "__main__":
    split_demo()

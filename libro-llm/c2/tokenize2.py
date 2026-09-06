import tiktoken

example_file = "the-verdict.txt"


def get_file_content(input_file: str) -> str:
    with open(input_file, "r", encoding="utf-8") as ifile:
        return ifile.read()


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")

    text = """This establishes the main-axis, thus defining the direction flex items 
    are placed in the flex container. 
    Flexbox is (aside from optional wrapping) a single-direction layout concept. 
    Think of flex items as primarily laying out either in horizontal rows or vertical columns."""

    tokenized = tokenizer.encode(text)

    decoded = tokenizer.decode(tokenized)

    print(text)
    print(tokenized)
    print(decoded)

example_file = "the-verdict.txt"

import re
 
def tokenize_file(input_file:str)->list[str]:
    with open(input_file, "r", encoding="utf-8") as ifile:
        corpus = ifile.read()
        splitted = re.split(r"([,.;:?!'\"(\\/)]|--|\s)", corpus)
        tokenized = [token for token in splitted if token]
        return tokenized

def create_embedding_dict(tokenized: list[str]):
    print("creating set...")
    tokens = set(tokenized)
    print("creating dict...")
    return {
        token : index
        for token, index in enumerate(tokens)
    }

if __name__ == '__main__':
    tokenized = tokenize_file(example_file)
    
    embedding_dict = create_embedding_dict(tokenized)

    for k, v in embedding_dict.items():
        print(f"{k}: {v}")
 
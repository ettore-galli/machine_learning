from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

print(type(tokenizer))

INPUT_DATA = [
    "input_ids: numerical representations of your tokens",
    "token_type_ids: these tell the model which part of the input is sentence A and which is sentence B (discussed more in the next section)",
    "attention_mask: this indicates which tokens should be attended to and which should not (discussed more in a bit)",
]

# encoded_input = tokenizer(*INPUT_DATA, padding=True, truncation=True, max_length=200)

tokens = tokenizer.tokenize(INPUT_DATA[0])
print(tokens)

ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

INPUT_DATA = [
    "input_ids: numerical representations of your tokens",
    "token_type_ids: these tell the model which part of the input is sentence A and which is sentence B (discussed more in the next section)",
    "attention_mask: this indicates which tokens should be attended to and which should not (discussed more in a bit)",
]

model_inputs = tokenizer(INPUT_DATA, padding=True, truncation=True, return_tensors="pt")

result = model(**model_inputs)

print(result.logits)

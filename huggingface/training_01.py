from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# This is new
batch["labels"] = torch.tensor([1, 1])


# optimizer = AdamW(model.parameters())
# loss = model(**batch).loss
# loss.backward()
# optimizer.step()


# inputs = tokenizer("This is the first sentence.", "This is the second one.", "third sentence.")
# inputs

raw_datasets = load_dataset("glue", "mrpc")

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# batch = data_collator(tokenized_datasets)
training_args = TrainingArguments("test-trainer")

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
)

trainer.train()

predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)
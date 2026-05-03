import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from models.general_llm_base import GeneralLLMBase


class ClassifierNLILLM(GeneralLLMBase):
    def __init__(self, logger):
        super().__init__("facebook/bart-large-mnli", logger)

    def do_perform(self, prompt: str, **kwargs) -> str:
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        hypothesis = "Given sentence requires to perform an arithmetic operation"

        tokenized_inputs = tokenizer.encode(
            prompt,
            hypothesis,
            return_tensors="pt",
        )

        inputs = self.prepare_input_for_device(
            tokenized_inputs=tokenized_inputs, device=self.device
        )

        with torch.no_grad():
            outputs = model(inputs, **kwargs)
            probs = outputs.logits.softmax(dim=1)

        return {
            label: probs[0][result_id].item()
            for result_id, label in model.config.id2label.items()
        }

    def perform(self, prompt: str, **kwargs) -> str:
        return self.do_perform(
            prompt,
            **{
                **dict(
                    max_new_tokens=512,
                    repetition_penalty=1.5,
                    do_sample=True,
                    top_k=50,
                    length_penalty=1.0,
                    no_repeat_ngram_size=3,
                    truncation_strategy="only_first",
                ),
                **kwargs,
            },
        )

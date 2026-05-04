import torch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
)

from models.general_llm_base import GeneralLLMBase


class ClassifierLLM(GeneralLLMBase):
    def __init__(self, logger):
        super().__init__("distilbert-base-uncased-finetuned-sst-2-english", logger)

    def do_perform(self, prompt: str, **kwargs) -> str:
        tokenizer = DistilBertTokenizer.from_pretrained(self.model_id)
        model = DistilBertForSequenceClassification.from_pretrained(
            self.model_id,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        inputs = tokenizer(self.prepare_propmpt(prompt), return_tensors="pt")

        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        generation_params = {**dict(num_beams=4), **kwargs}

        with torch.no_grad():
            outputs = model(**inputs, **generation_params)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return {
            label: predictions[0][result_id].item()
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
                ),
                **kwargs,
            },
        )

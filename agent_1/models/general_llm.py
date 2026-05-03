from typing import Any

from models.general_llm_base import GeneralLLMBase


class GeneralLLM(GeneralLLMBase):
    def perform(self, prompt: str, **kwargs: Any) -> str:
        return super().perform(
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

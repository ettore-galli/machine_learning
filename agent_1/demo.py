import logging

logger = logging.getLogger(__name__)

from models.llm_text_model import GeneralLLM


def demo():
    gen_llm = GeneralLLM(model_id="google/flan-t5-base", logger=logger)


if __name__ == "__main__":
    demo()

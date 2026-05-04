import logging

from models.general_llm import GeneralLLM

logger = logging.getLogger(__name__)


def demo():
    gen_llm = GeneralLLM(model_id="google/flan-t5-small", logger=logger)
    response = gen_llm.perform("""Answer the following question:
        How are emnedding used as input of transformers?
        Assume knowledge of what an embedding is
        Assume LLM as context
        Give a detailed answer
        """)
    print("-" * 80)
    print(response)
    print("-" * 80)


if __name__ == "__main__":
    demo()

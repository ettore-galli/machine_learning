import logging

from models.classifier import ClassifierLLM

logger = logging.getLogger(__name__)


def demo():
    gen_llm = ClassifierLLM(logger=logger)
    response = gen_llm.perform("""Answer the following question:
        How are emnedding used as input of transformers?
        Assume knowledge of what an embedding is
        Assume LLM as context
        Give a detailed answer
        """)
    print("-" * 80)
    print(response)
    print("-" * 80)


if __name__ in ["__main__"]:
    demo()

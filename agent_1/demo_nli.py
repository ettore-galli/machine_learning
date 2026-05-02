import logging

from models.classifier_nli import ClassifierNLILLM

logger = logging.getLogger(__name__)


def demo():
    gen_llm = ClassifierNLILLM(logger=logger)
    response = gen_llm.perform("""Add 75 to 47
        """)
    print("-" * 80)
    print(response)
    print("-" * 80)


if __name__ in ["__main__"]:
    demo()

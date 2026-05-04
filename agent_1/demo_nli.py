import logging
from typing import List

from models.classifier_nli import ClassifierNLILLM

logger = logging.getLogger(__name__)


def demo():
    gen_llm = ClassifierNLILLM(logger=logger)

    test_sentences: List[str] = [
        """Add 75 to 4""",
        "subtract 45 from 97",
        "What a bad weather",
        "I am 53 years old",
    ]
    for test_sentence in test_sentences:
        response = gen_llm.perform(test_sentence)

        print("-" * 80)
        print(test_sentence)
        print(response)
        print("-" * 80)


if __name__ in ["__main__"]:
    demo()

import logging
from typing import List

from models.classifier_model_base import ClassifierResultType
from models.classifier_model_tools import retrieve_response_ranking
from models.classifier_nli import ClassifierNLILLM

logger = logging.getLogger(__name__)


def format_case(hypotesis: str, response: ClassifierResultType):
    ranking = retrieve_response_ranking(response=response)
    length = 40
    return f"{hypotesis[:length].ljust(length+1)}: {ranking[0][0].upper()} ({100*ranking[0][1]} % )"


def demo():
    gen_llm = ClassifierNLILLM(logger=logger)

    test_sentences: List[str] = [
        ("The quick brown fox jumps over the lazy dog"),
        ("""
            When I was younger, so much younger than today,
            I never needed anybody's help in any way,
            But now these days are gone and I'm not so self assured,
            Now I find I've changed my mind, I've opened up the doors.

            Help me if you can, I'm feeling down,
            And I do appreciate you being 'round,
            Help me get my feet back on the ground,
            Won't you please, please help me?
            """),
        ("""
            This is Titanic speaking just hit an Iceberg the ship is sinking we need immediate assistance
            """),
    ]

    hypotheses: List[str] = [
        "This text is a request for help",
        "This is a real request for help",
        "This is a metaphoric request for help",
        "This is a song or poem",
        "This is a haiku",
    ]

    for test_sentence in test_sentences:
        print("-" * 80)
        print(test_sentence)
        for hypotesis in hypotheses:

            response = gen_llm.perform(test_sentence, hypotesis)
            print(format_case(hypotesis=hypotesis, response=response))


if __name__ in ["__main__"]:
    demo()

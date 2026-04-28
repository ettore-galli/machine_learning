import logging

logger = logging.getLogger(__name__)

from agent_1.models.general_llm import GeneralLLM


def demo():
    gen_llm = GeneralLLM(model_id="google/flan-t5-large", logger=logger)
    response = gen_llm.perform(
        """Answer the following question:
        How are emnedding used as input of transformers?
        Assume knowledge of what an embedding is
        Assume LLM as context
        Give a technical answer
        """
    )
    print(response)


if __name__ == "__main__":
    demo()

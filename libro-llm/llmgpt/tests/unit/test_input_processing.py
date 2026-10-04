from llmgpt.input_processing import produce_input_embeddings
from torch import equal, tensor


def test_produce_input_embeddings():
    assert equal(
        produce_input_embeddings(["Hello,", "The pen is on"]),
        tensor([[15496, 11, 0, 0], [464, 3112, 318, 319]]),
    )

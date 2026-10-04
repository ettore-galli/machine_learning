import torch
from llmgpt.model import GptNN
from llmgpt.config import get_config
from llmgpt.input_processing import produce_input_embeddings


def try_model_basic_inference_behaviour():
    torch.manual_seed(123)
    model = GptNN(config=get_config())
    inputs = produce_input_embeddings(["The Quick Brown Fox", "The pen is"])
    logits = model(inputs)
    print(logits)


if __name__ == '__main__':
    try_model_basic_inference_behaviour()
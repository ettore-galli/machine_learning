from typing import cast

import matplotlib.pyplot as plt
import torch
from llmgpt.config import get_config
from llmgpt.model import GptNN
from torch import nn


def load_model() -> torch.nn.Module:
    model = GptNN(config=get_config())
    config = get_config()
    model.load_state_dict(torch.load(config.model_weights_file))

    return model


def evaluate():
    model = load_model()
    model.eval()
    with torch.no_grad():
        print("\n-----\n")
        example_data = torch.Tensor([[1, -4], [1, 1], [1, 3], [1, 5]])
        for example in example_data:
            result = model.forward(example)
            print(f"{example} => {result} => {torch.sigmoid(result)}")

        show_model(model=model)


def show_model(model: torch.nn.Module) -> None:
    net = cast(nn.Sequential, model.net)

    plt.imshow(
        net[2].weight.detach().numpy(),  # pyright: ignore[reportCallIssue]
        cmap="viridis",
    )
    plt.colorbar()
    plt.title("Pesi layer 1")
    plt.show()


if __name__ == "__main__":

    evaluate()

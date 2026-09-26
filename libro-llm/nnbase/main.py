import torch

from model import SimpleNN
from common import get_config


def load_model() -> torch.nn.Module:
    model = SimpleNN()
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


if __name__ == "__main__":

    evaluate()

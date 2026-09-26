import torch

from dataset import X, y
from model import model
from train import train_model


def evaluate():
    print("\n-----\n")
    example_data = torch.Tensor([[1, -3], [1, 3], [1, 2]])
    print(f"{example_data} => {model.forward(example_data)}")


if __name__ == "__main__":
    train_model(model=model, X=X, y=y)

    evaluate()

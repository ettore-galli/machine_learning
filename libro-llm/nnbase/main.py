import torch

from dataset import X, y
from model import model
from train import train_model


def evaluate():
    model.eval()
    with torch.no_grad():
        print("\n-----\n")
        example_data = torch.Tensor([[1, -4], [1, 1], [1, 3], [1, 5]])
        for example in example_data:
            result = model.forward(example)
            print(f"{example} => {result} => {torch.sigmoid(result)}")


if __name__ == "__main__":
    train_model(model=model, X=X, y=y)

    evaluate()

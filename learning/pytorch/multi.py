from typing import cast
import torch
from torch.utils.data import Dataset, DataLoader


class NeuralExample(torch.nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        num_hidden_1: int = 30,
        num_hidden_2: int = 20,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, num_hidden_1),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden_1, num_hidden_2),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden_2, num_outputs),
        )

    def print_info(self):
        print(type(self.layers))
        print(cast(torch.nn.modules.linear.Linear, self.layers[0]).weight)

    def forward(self, x):
        logit = self.layers(x)
        return logit


class DataSet(Dataset):
    def __init__(self, features: torch.tensor, labels: torch.tensor):
        self.features: torch.tensor = features
        self.labels: torch.tensor = labels

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return cast(torch.tensor, self.labels).shape[0]


def nn_demo():
    ne = NeuralExample(2, 1)
    input = torch.tensor([1.0, 1.0])
    with torch.no_grad():
        out = ne(input)
        print(out)


def nn_demo_2():
    x_train = torch.tensor(
        [
            [1.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.9],
            [0.0, 0.0],
            [0.85, 0.0],
            [0.0, 0.9],
        ]
    )
    y_train = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.99, 0.95])

    x_test = torch.tensor(
        [
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )
    y_test = torch.tensor([0.0, 1.0])

    train_ds = DataSet(x_train, y_train)
    test_ds = DataSet(x_test, y_test)

    batch_size = 2

    train_loader = DataLoader(
        dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        dataset=test_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )

    for idx, data in enumerate(train_loader):
        print(f"{idx}: {data}")


if __name__ == "__main__":
    nn_demo_2()

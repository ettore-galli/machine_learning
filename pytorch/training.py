from typing import Tuple, cast

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class NeuralNetwork(torch.nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        num_hidden_1: int,
        num_hidden_2: int,
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


def get_training_data() -> Tuple[DataLoader, DataLoader]:
    x_train = torch.tensor(
        [
            [1.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            #
            [0.8, 0.8],
            [0.2, 0.2],
            [0.8, 0.2],
            [0.2, 0.8],
            #
            [0.9, 0.9],
            [0.1, 0.1],
            [0.1, 0.9],
            [0.9, 0.1],
            #
            [0.7, 0.7],
            [0.3, 0.3],
            [0.7, 0.2],
            [0.3, 0.3],
        ]
    )
    y_train = torch.tensor(
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1], dtype=torch.long
    )

    x_test = torch.tensor(
        [
            [0.8637452, 0.9453],
            [0.026345, 0.98987],
        ]
    )
    y_test = torch.tensor([0, 1], dtype=torch.long)

    train_ds = DataSet(x_train, y_train)
    test_ds = DataSet(x_test, y_test)

    batch_size = y_train.shape[0]

    train_loader = DataLoader(
        dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        dataset=test_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )

    return train_loader, test_loader


def train_network(
    model: NeuralNetwork,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

    for epoch in range(num_epochs):
        model.train()

        for batch_idx, (features, labels) in enumerate(train_loader):

            outputs = model(features)

            loss = F.cross_entropy(input=outputs, target=labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


def use_network(model: NeuralNetwork):
    model.eval()

    x_real = torch.tensor([[0.9, 0.1], [0.7, 0.1], [0.1, 0.1], [0.8, 0.8]])

    raw = model(x_real)

    print(raw)

    torch.set_printoptions(sci_mode=False)
    probabilities = torch.softmax(raw, dim=1)

    for example, probs in zip(x_real, probabilities):
        print(f"{example}, => {probs}")


def training_main():
    torch.manual_seed(123)

    model = NeuralNetwork(num_inputs=2, num_outputs=2, num_hidden_1=8, num_hidden_2=8)
    train_loader, test_loader = get_training_data()

    train_network(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=1000,
    )

    use_network(model)


if __name__ == "__main__":
    training_main()

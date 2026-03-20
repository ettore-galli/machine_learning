from typing import Iterable, Tuple, cast

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


def get_training_data(device: str = "cpu") -> Tuple[DataLoader, DataLoader]:
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
    ).to(device=device)
    y_train = torch.tensor(
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1], dtype=torch.long
    ).to(device=device)

    x_test = torch.tensor(
        [
            [0.8637452, 0.9453],
            [0.026345, 0.98987],
        ]
    ).to(device=device)
    y_test = torch.tensor([0, 1], dtype=torch.long).to(device=device)

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


def tensors_to_device(
    tensors: Iterable[torch.tensor], device: str
) -> Tuple[torch.tensor, ...]:
    tensor: torch.tensor
    return (tensor.to(device) for tensor in tensors)


def train_network(
    model: NeuralNetwork,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    device: str,
):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

    for _ in range(num_epochs):
        model.train()

        for features_cpu, labels_cpu in train_loader:
            (features, labels) = tensors_to_device(
                (features_cpu, labels_cpu), device=device
            )

            outputs = model(features)

            loss = F.cross_entropy(input=outputs, target=labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


def use_network(model: NeuralNetwork, device: str):
    model.eval()

    x_real = torch.tensor([[0.9, 0.1], [0.7, 0.1], [0.1, 0.1], [0.8, 0.8]]).to(
        device=device
    )

    raw = model(x_real)

    print(raw)

    torch.set_printoptions(sci_mode=False)
    probabilities = torch.softmax(raw, dim=1)

    for example, probs in zip(x_real, probabilities):
        print(f"{example}, => {probs.tolist()}")


def training_main(model_name: str, device: str):

    torch.manual_seed(123)

    model_file = f"./saved-models/{model_name}.pth"

    model = NeuralNetwork(
        num_inputs=2, num_outputs=2, num_hidden_1=8, num_hidden_2=8
    ).to(device=device)
    train_loader, test_loader = get_training_data(device=device)

    train_network(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=1000,
        device=device,
    )

    torch.save(model.state_dict(), model_file)

    use_network(model, device=device)


def get_device() -> str:
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"DEVICE: *** {device} ***")
    return device


if __name__ == "__main__":
    training_main("example-1", device=get_device())

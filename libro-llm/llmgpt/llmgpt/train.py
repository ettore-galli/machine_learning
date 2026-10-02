import torch
from torch import nn

from llmgpt.common import Config


def train_model(
    config: Config,
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    number_of_epochs: int = 200,
):
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    model.train()

    for epoch in range(number_of_epochs):
        # Forward
        logits = model(X).squeeze()
        loss = loss_fn(logits, y)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Update
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), config.model_weights_file)

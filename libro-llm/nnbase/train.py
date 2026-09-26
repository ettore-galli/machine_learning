import torch
import torch.nn as nn


def train_model(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor, number_of_epochs: int = 200
):
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

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

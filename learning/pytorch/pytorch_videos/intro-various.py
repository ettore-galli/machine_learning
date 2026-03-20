import torch


def example_function(x: torch.tensor, slope: float) -> torch.tensor:
    return (slope * x).tanh()


def diff():
    torch.manual_seed(123)

    for f in [0.01, 0.1, 1, 2, 5, 10]:

        x = torch.ones([1, 10], requires_grad=True)

        alfa = f * x

        beta = alfa.tanh()

        loss = beta.sum()

        loss.backward()

        print(f"f={f}; grad={x.grad.mean()}")


if __name__ == "__main__":
    diff()

# Dataset sintetico
import matplotlib.pyplot as plt
import torch

N = 500
X = torch.randn(N, 2)
true_w = torch.tensor([1.0, -1.0])
true_b = 0

# y = 1 se w·x + b > 0
y = (X @ true_w + true_b > 0).float()


if __name__ == "__main__":
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()

# Dataset sintetico
import matplotlib.pyplot as plt
import torch

N = 500
X = torch.randn(N, 2)
true_w = torch.tensor([1.0, -1.0])
true_b = 0

# y = 1 se w·x + b > 0
y = (X @ true_w + true_b > 0).float()


def plotdiv(n: int, d: int) -> list[str]:
    def makeline(stars: int) -> str:
        return "*" * n + "\n"

    residual = n
    plot = []

    while residual > n:
        plot.append(makeline(stars=d))
        residual = residual - d

    plot.append(makeline(stars=residual))

    return plot


if __name__ == "__main__":
    # plt.scatter(X[:, 0], X[:, 1], c=y)
    # plt.show()

    # for line in plotdiv(221, 17):
    #     print(line)
    print(plotdiv(221, 17))
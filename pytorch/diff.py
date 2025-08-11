import torch
import torch.nn.functional as F
from torch.autograd import grad

if __name__ == "__main__":

    y = torch.tensor([1.0])
    x1 = torch.tensor([1.1])
    w1 = torch.tensor([2.2], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)
    z = x1 * w1 + b
    a = torch.sigmoid(z)

    loss = F.binary_cross_entropy(a, y)

    grads = [grad(loss, w1, retain_graph=True), grad(loss, b, retain_graph=True)]

    print(y, x1, w1, b, z, a, loss)
    print(grads)

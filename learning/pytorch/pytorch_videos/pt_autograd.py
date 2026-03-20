# autograd

import torch

from device_utils import get_available_device


def grad_base():

    input = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    result = (input @ torch.tensor([1.0, 1.0, 1.0])).square().sum()
    result.backward()
    grad = input.grad

    return "grad_base", [("result", result), ("grad", grad)]


def grad_example():
    results = []

    for b in [1.0, 2.0]:
        input = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result = (input @ torch.tensor([1.0, b, 1.0])).square().sum()
        result.backward()
        grad = input.grad
        results.append((f"b={b}", (result, grad)))

    return "grad_example", results


def autograd_deep_dive():
    for title, result in [
        grad_base(),
        grad_example(),
    ]:
        print_title = f"\n*** {title} ***"

        print("_" * len(print_title))
        print(print_title)
        print("-" * len(print_title))

        if type(result) in [list, tuple]:
            for descr, value in result:
                print(f"\n  {descr.ljust(20)}: {value}")
        else:
            print(result)


if __name__ == "__main__":
    autograd_deep_dive()

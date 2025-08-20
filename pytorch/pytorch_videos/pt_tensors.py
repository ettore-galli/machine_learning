# TENSORS

import torch

from device_utils import get_available_device


def get_device():
    candidates = [torch.cuda, torch.backends.mps]


def broadcasting():
    alfa = torch.tensor([[1, 2], [7, 11]])
    beta = torch.tensor([[3, 3]])

    return "broadcasting", alfa * beta


def tensor_math():
    alfa = torch.tensor([[1, 2], [7, 11]])
    alfa2 = torch.tensor([[1, 7], [7, 11]])

    return "tensor_math", [
        ("EQ", alfa.eq(alfa2)),
        ("MAX", alfa.max()),
        ("UNIQUE", torch.tensor([[[1, 7]], [[7, 11]]]).unique()),
        ("MEAN", torch.tensor([[12, 14], [101, 103]]).float().mean(dim=1)),
        ("CROSS", torch.cross(torch.tensor([1.0, 0, 0]), torch.tensor([0, 1.0, 0]))),
    ]


def tensor_inplace():
    results = []
    alfa = torch.tensor([1, 2, 3])
    results.append(("alfa pre", alfa.__str__()))
    results.append(("square", alfa.square_()))
    results.append(("alfa post", alfa))

    return "tensor_inplace", results


def tensor_op_out():
    results = []
    alfa = torch.tensor([[1.0, 2, 3]])
    beta = torch.tensor([1.0, 2, 3])
    gamma = torch.zeros([1])
    gamma_id_pre = id(gamma)
    gamma = torch.matmul(alfa, beta.T, out=gamma)
    gamma_id_post = id(gamma)
    results.append(("gamma_id_pre", gamma_id_pre))
    results.append(("gamma", gamma))
    results.append(("gamma_id_post", gamma_id_post))

    return "tensor_op_out", results


def tensor_device():
    alfa = torch.tensor([1, 2, 3], device=get_available_device())
    return ("tensor_device", alfa.device)


def tensor_dimensions():
    alfa = torch.tensor([1, 2, 3])
    beta = torch.tensor([[[1], [2], [3]]])
    return (
        "tensor_dimensions",
        [
            ("alfa", alfa),
            ("alfa unsqueeze 0", alfa.unsqueeze(dim=0)),
            ("alfa unsqueeze 1", alfa.unsqueeze(dim=1)),
            ("beta", beta),
            ("alfa squeeze 0", beta.squeeze(dim=0)),
            ("alfa squeeze 1", beta.squeeze(dim=1)),
            ("alfa squeeze 2", beta.squeeze(dim=2)),
        ],
    )


def tensor_reshape():
    alfa = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

    
    return (
        "tensor_dimensions",
        [
            ("alfa", alfa),
            ("alfa reshape", alfa.reshape(12)),
            ("alfa reshape", alfa.reshape(4, 3)),
            ("alfa reshape", alfa.reshape(3, 4)),
            ("alfa reshape", alfa.reshape(2, 6)),
            ("alfa reshape", alfa.reshape(6, 2)),
        ],
    )


def tensors_deep_dive():
    for title, result in [
        broadcasting(),
        tensor_math(),
        tensor_inplace(),
        tensor_op_out(),
        tensor_device(),
        tensor_dimensions(),
        tensor_reshape(),
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
    tensors_deep_dive()

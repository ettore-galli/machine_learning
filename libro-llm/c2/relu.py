import torch
from torch import nn

from shortcuts import t


def main():
    torch.manual_seed(123)
    batch_example = torch.randn(2, 5)

    linear_layer = nn.Linear(5, 6)
    relu_layer = nn.ReLU()

    layer = nn.Sequential(linear_layer, relu_layer)
    
    out_overall = layer(batch_example)

    print("\n\nbatch_example")
    print(batch_example)

    out_linear = linear_layer(batch_example)
    print("\n\nout_linear")
    print(out_linear)

    out_relu = relu_layer(out_linear)
    print("\n\nout_relu")
    print(out_relu)

    print("\n\nout_overall")
    print(out_overall)


if __name__ == "__main__":
    main()

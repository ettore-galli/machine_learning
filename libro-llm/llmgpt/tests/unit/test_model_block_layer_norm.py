import torch
from llmgpt.model_blocks import LayerNorm


def test_layer_norm_basic_behaviour():
    input = torch.Tensor([0.3000, 0.4000, 1.1000, 5.0000, 3.1415])

    layer_norm = LayerNorm(5)
    assert torch.allclose(
        layer_norm.forward(input),
        torch.tensor([-0.9276, -0.8727, -0.4881, 1.6548, 0.6336]),
        atol=0.0001,
    )

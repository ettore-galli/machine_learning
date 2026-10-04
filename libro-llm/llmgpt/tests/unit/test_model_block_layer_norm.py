import torch
from llmgpt.model_blocks import LayerNorm


def test_layer_norm_basic_behaviour():
    input = torch.Tensor([0.3000, 0.4000, 1.1000, 5.0000, 3.1415])

    layer_norm = LayerNorm(5)
    assert torch.allclose(
        layer_norm.forward(input),
        torch.tensor([-0.8297, -0.7806, -0.4365, 1.4801, 0.5667]),
        atol=0.0001,
    )

from torch import nn

from llmgpt.config import GPTConfig


class TransformerBlock(nn.Module):
    def __init__(self, gptr_config: GPTConfig):
        super().__init__()

    def forward(self, x):
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()

    def forward(self, x):
        return x

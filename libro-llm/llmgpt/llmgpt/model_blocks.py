import torch
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
        self.eps: float = eps
        self.scale = nn.Parameter(torch.ones(normalized_shape))
        self.offset = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, input: torch.Tensor):
        avg: torch.Tensor = torch.mean(input, dim=-1, keepdim=True)
        var: torch.Tensor = torch.var(input, dim=-1, keepdim=True, unbiased=False)
        norm: torch.Tensor = (input - avg) / torch.sqrt(
            torch.where(var == 0, self.eps, var)
        )
        return self.scale * norm + self.offset

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

    def forward(self, input: torch.Tensor):
        avg: torch.Tensor = torch.mean(input, dim=-1, keepdim=True)
        var: torch.Tensor = torch.var(input, dim=-1, keepdim=True)
        return (input - avg) / torch.sqrt(torch.where(var == 0, self.eps, var))

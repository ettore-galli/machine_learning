from dataclasses import dataclass

from llmgpt.dataclassdict import DataClassDict


@dataclass(frozen=True)
class GPTConfig(DataClassDict):
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    drop_rate: float
    qkv_bias: bool


@dataclass(frozen=True)
class Config(DataClassDict):
    model_weights_file: str
    gpt_config: GPTConfig


def get_gpt_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=50257,
        context_length=1024,
        emb_dim=768,
        n_heads=12,
        n_layers=12,
        drop_rate=0.1,
        qkv_bias=False,
    )


def get_config() -> Config:
    gpt_config = get_gpt_config()
    return Config(model_weights_file="parameters/SimpleNN.pth", gpt_config=gpt_config)

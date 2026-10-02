from dataclasses import dataclass

from llmgpt.dataclassdict import DataClassDict


@dataclass(frozen=True)
class Config(DataClassDict):
    model_weights_file: str


def get_config() -> Config:
    return Config(model_weights_file="parameters/SimpleNN.pth")

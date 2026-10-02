from dataclasses import dataclass


@dataclass
class Config:
    model_weights_file: str


def get_config() -> Config:
    return Config(model_weights_file="parameters/SimpleNN.pth")

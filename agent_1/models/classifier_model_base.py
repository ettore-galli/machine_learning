from dataclasses import dataclass
from typing import Dict, Protocol

KeywordArgsType = Dict[str, int | float | str]


@dataclass(frozen=True)
class Issue:
    message: str
    success: bool = False


class ModelClassifierNLIProtocol(Protocol):
    def __call__(self, text: str, text_pair: str) -> Dict[str, float]: ...


MANDATORY_ENVVARS = [
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_DATASETS_CACHE",
    "TRANSFORMERS_CACHE",
    "DIFFUSERS_CACHE",
]

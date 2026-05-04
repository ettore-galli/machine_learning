from dataclasses import dataclass
from typing import Dict

KeywordArgsType = Dict[str, int | float | str]


@dataclass(frozen=True)
class Issue:
    message: str
    success: bool = False


MANDATORY_ENVVARS = [
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_DATASETS_CACHE",
    "TRANSFORMERS_CACHE",
    "DIFFUSERS_CACHE",
]

from dataclasses import dataclass
from typing import Dict, List, Protocol, Union

KeywordArgsValueType = Union[int, float, str]
KeywordArgsType = Dict[str, KeywordArgsValueType]

ClassifierResultType = Dict[str, float]


@dataclass(frozen=True)
class Issue:
    message: str
    success: bool = False


class ModelClassifierNLIProtocol(Protocol):
    def __call__(
        self, text: str, text_pair: str, **kwargs: KeywordArgsValueType
    ) -> Dict[str, float]: ...


KNOWN_MANDATORY_ENVVARS: List[str] = [
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_DATASETS_CACHE",
    "TRANSFORMERS_CACHE",
    "DIFFUSERS_CACHE",
]

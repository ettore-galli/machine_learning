from dataclasses import dataclass
from typing import Dict

KeywordArgsType = Dict[str, int | float | str]


@dataclass(frozen=True)
class Issue:
    message: str
    success: bool = False

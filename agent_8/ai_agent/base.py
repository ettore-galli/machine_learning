from abc import ABC, abstractmethod
from typing import Protocol


class ResponseDisplayer(Protocol):
    def __call__(self, *, segment: str) -> None: ...


class AIAgentBase(ABC):
    @abstractmethod
    def generate(
        self,
        user_prompt: str,
        system_prompt: str,
        user_prompt_marker: str,
        response_displayer: ResponseDisplayer,
    ) -> str: ...

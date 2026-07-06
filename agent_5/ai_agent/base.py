from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, TypedDict
from langchain.messages import HumanMessage

from langchain.agents.middleware.types import (
    InputAgentState,
)

AgentStateMessagesType = List[Dict[str, Any]]


@dataclass(frozen=True)
class AgentSettings:

    llama_cpp_server_url: str
    llama_cpp_server_model: str

    @staticmethod
    def load() -> AgentSettings:
        """
        Carica le variabili d'ambiente e costruisce l'oggetto settings.
        """
        import os

        return AgentSettings(
            llama_cpp_server_url=os.getenv("LLAMA_CPP_SERVER_URL", "").strip(),
            llama_cpp_server_model=os.getenv("MODEL_QWEN253B_NAME", "").strip(),
        )


class AgentState(TypedDict):
    messages: AgentStateMessagesType


def get_initial_agent_state(initial_user_prompt: str) -> AgentState:
    return AgentState(messages=[{"role": "user", "content": initial_user_prompt}])


def build_agent_input(initial_user_prompt: str) -> InputAgentState:
    return InputAgentState({"messages": [HumanMessage(content=initial_user_prompt)]})


def extract_response_message(response: Dict) -> str:
    return response["messages"][-1].content

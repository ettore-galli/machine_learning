from typing import Any

from langchain.agents.middleware.types import (
    AgentState,
    ContextT,
    InputAgentState,
    OutputAgentState,
)

from langgraph.graph.state import CompiledStateGraph

AgentType = CompiledStateGraph[
    AgentState[Any], ContextT, InputAgentState, OutputAgentState[Any]
]

from typing import Any
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime


class DebugMiddleware(AgentMiddleware):
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(state)

        return None

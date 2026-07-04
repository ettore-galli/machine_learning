import json
from typing import Any, Protocol
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.messages import AIMessage, AnyMessage
from langgraph.runtime import Runtime


class ToolCallMiddleware(AgentMiddleware):
    def __init__(self):
        super().__init__()

    def build_standard_tool_call(self, message: AIMessage) -> AIMessage:
        raw_json = message.content
        print(f"\n{'*'*70}\n{raw_json}\n{'*'*70}\n")

        tool_data = json.loads(raw_json)

        tool_calls = [
            {
                "name": tool_data["name"],
                "args": tool_data.get("arguments") or tool_data.get("args", {}),
                "id": tool_data.get("id", f"call_{hash(raw_json) % 10000}"),
            }
        ]
        return AIMessage(
            content=message.content,
            tool_calls=tool_calls,
            response_metadata=message.response_metadata,
        )

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        state["messages"].append(self.build_standard_tool_call(state["messages"][-1]))
        return

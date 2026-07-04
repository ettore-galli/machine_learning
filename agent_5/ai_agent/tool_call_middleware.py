import json
import re
from typing import Any, cast
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.messages import AIMessage, AnyMessage
from langgraph.runtime import Runtime


class ToolCallMiddleware(AgentMiddleware):
    def __init__(self):
        super().__init__()
        self.tool_request_re = r'{"tool":{"name":.*}'

    def get_tool_request_from_message(self, message_content: str) -> str | None:
        match = re.search(
            self.tool_request_re, message_content, re.DOTALL | re.IGNORECASE
        )
        if match:
            return match.group(0)
        return None

    def normalize_message_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(str(chunk) for chunk in content)
        return str(content)

    def build_standard_tool_call(self, message: AIMessage) -> AIMessage | None:

        if not message.content:
            return None

        message_content: str = self.normalize_message_content(message.content)
        candidate_tool_call: str | None = self.get_tool_request_from_message(
            message_content=message_content
        )

        if candidate_tool_call:

            tool_data = json.loads(candidate_tool_call)

            tool_calls = [
                {
                    "name": tool_data["tool"]["name"],
                    "args": tool_data.get("tool", {}).get("arguments")
                    or tool_data.get("args", {}),
                    "id": tool_data.get(
                        "id", f"call_{hash(candidate_tool_call) % 10000}"
                    ),
                }
            ]
            return AIMessage(
                content=message_content,
                tool_calls=tool_calls,
                response_metadata=message.response_metadata,
            )

        return message

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if len(state["messages"]) == 0:
            return cast(dict[str, Any], state)

        last_message: AnyMessage = state["messages"][-1]

        tool_call_message: AIMessage | None = (
            self.build_standard_tool_call(message=last_message)
            if isinstance(last_message, AIMessage)
            else None
        )
        if tool_call_message and tool_call_message.tool_calls:
            state["messages"].append(tool_call_message)

        return cast(dict[str, Any], state)

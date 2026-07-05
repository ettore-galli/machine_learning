import json
import re
from typing import Any, cast
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.messages import AIMessage, AnyMessage
from langgraph.runtime import Runtime


class ToolCallMiddleware(AgentMiddleware):
    def __init__(self):
        super().__init__()
        self.tool_request_re_cases = [
            r'{"tool":{"name":.*}',
            r"<tool_call>(.*)</tool_call>",
        ]

    def get_tool_request_from_message(self, message_content: str) -> str | None:
        for tool in [
            self.get_tool_request_from_message_by_regexp(
                message_content=message_content, regex=re_case
            )
            for re_case in self.tool_request_re_cases
        ]:
            if tool:
                return tool
        return None

    @staticmethod
    def get_tool_request_from_message_by_regexp(
        message_content: str, regex: str
    ) -> str | None:
        match = re.search(regex, message_content, re.DOTALL | re.IGNORECASE)
        if match:
            candidates = [*match.groups()[::-1], match.group(0)]
            return next((candidate for candidate in candidates if candidate)).strip()
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

            tool_call_source = json.loads(candidate_tool_call)
            tool_data = (
                tool_call_source["tool"]
                if "tool" in tool_call_source
                else tool_call_source
            )

            tool_calls = [
                {
                    "name": tool_data["name"],
                    "args": tool_data.get("arguments") or tool_data.get("args", {}),
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

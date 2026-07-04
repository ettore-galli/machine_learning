import re
import json
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda


def qwen_tool_call_parser(message):
    """Parser robusto per formato Qwen <tool_call>"""
    if not isinstance(message, AIMessage):
        return message

    content = message.content or ""

    # Pattern 1: <tool_call> JSON </tool_call>
    text = str(content)
    match = re.search(
        r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL | re.IGNORECASE
    )
    if match:
        try:
            raw = match.group(1).strip()
            tool_data = json.loads(raw)

            tool_calls = [
                {
                    "name": tool_data["name"],
                    "args": tool_data.get("arguments") or tool_data.get("args", {}),
                    "id": tool_data.get("id", f"call_{hash(raw) % 10000}"),
                }
            ]

            return AIMessage(
                content=content,
                tool_calls=tool_calls,
                response_metadata=message.response_metadata,
            )
        except (json.JSONDecodeError, KeyError):
            pass  # fallback

    # Pattern 2: JSON diretto nel contenuto (senza tag)
    text = str(content)
    json_match = re.search(r'(\{.*?"name".*?"arguments".*?\})', text, re.DOTALL)
    if json_match:
        try:
            tool_data = json.loads(json_match.group(1))
            # ... stesso codice di sopra
        except (json.JSONDecodeError, KeyError):
            pass

    return message  # nessun tool call trovato


def fix_qwen_tools(state):
    if state.get("messages"):
        state["messages"][-1] = qwen_tool_call_parser(state["messages"][-1])
    return state


# Avvolgi con RunnableLambda
fix_qwen_tools_middleware = RunnableLambda(fix_qwen_tools)

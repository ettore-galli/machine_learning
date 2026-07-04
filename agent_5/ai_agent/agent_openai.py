from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from ai_agent.base import AgentSettings
from pydantic import SecretStr

from ai_agent.tool_call_middleware import ToolCallMiddleware
from ai_agent.tools import calculate_average

DEFAULT_NO_RESPONSE: str = "Dati insufficienti per una risposta significativa"

agent_settings: AgentSettings = AgentSettings.load()


def initialize_agent():
    model = ChatOpenAI(
        model=agent_settings.llama_cpp_server_model,  # es. "mistral"
        api_key=SecretStr("not-needed"),
        temperature=0,
        base_url=f"{agent_settings.llama_cpp_server_url}/v1",
    )

    agent = create_agent(
        model=model,
        tools=[calculate_average],
        middleware=[ToolCallMiddleware()],
        system_prompt="""You are a helpful assistant.

Your task is to respond, using a tool if and only if it is required and available.

1. If a tool is required AND available:
   - Output ONLY a JSON object in exactly this format:

     {"tool": {"name": "<tool_name>", "arguments": { ... }}}
     
   - Do NOT add text before or after the JSON.
   - Do NOT explain the tool call.
   - Do NOT include anything outside the JSON object.

2. If no tool is required OR no suitable tool exists:
   - Respond with a normal assistant message.
   - Do NOT output JSON.
   - Do NOT mention tools.

Rules:
- Never mix a normal answer with a tool call.
- Never invent tool names or arguments.
- Think step-by-step to decide if a tool is needed, but do NOT output your reasoning.

""",
    )

    return agent

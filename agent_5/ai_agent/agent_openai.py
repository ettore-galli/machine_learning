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

If a tool is required AND available:
    - Output ONLY the following JSON object:
      {"tool": {"name": "<tool_name>", "arguments": {"values": [...]}}}
    - The JSON must be the ONLY content in the message.
    - No text before or after.
    - No explanations.
    - No reasoning.
    - Do NOT invent tool names or arguments.

Otherwise respond normally without using tools

""",
    )

    return agent

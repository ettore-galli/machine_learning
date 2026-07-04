from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from ai_agent.base import AgentSettings
from pydantic import SecretStr

from ai_agent.debug_middleware import DebugMiddleware
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
        middleware=[DebugMiddleware()],
        system_prompt="""You are a helpful assistant.

When you need to use a tool, output ONLY the tool call in the following XML format, nothing else:

<tool_call>
{"name": "calculate_average", "arguments": {"values": [1, 2, 3]}}
</tool_call>

Do not add explanations, do not use JSON directly, do not use any other format.""",
    )

    return agent

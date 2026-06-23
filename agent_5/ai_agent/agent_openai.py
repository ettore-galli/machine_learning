from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from ai_agent.base import AgentSettings
from pydantic import SecretStr

DEFAULT_NO_RESPONSE: str = "Dati insufficienti per una risposta significativa"

agent_settings: AgentSettings = AgentSettings.load()


def initialize_agent():
    model = ChatOpenAI(
        model=agent_settings.llama_cpp_server_model,  # es. "mistral"
        api_key=SecretStr("not-needed"),
        temperature=0,
        base_url=f"{agent_settings.llama_cpp_server_url}/v1",
    )

    agent = create_agent(model=model)

    return agent

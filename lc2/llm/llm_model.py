from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from llm.base import LM_STUDIO_HOST, LM_STUDIO_MODEL
from llm.tools import get_downloads_directory, superbeta

llm_model = ChatOpenAI(
    base_url=LM_STUDIO_HOST,
    model=LM_STUDIO_MODEL,
    api_key=lambda: "no-key",
)

llm_agent = create_agent(
    model=llm_model,
    tools=[get_downloads_directory, superbeta],
    checkpointer=InMemorySaver(),
)

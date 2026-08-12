from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from llm.tools import get_downloads_directory
from llm.base import OLLAMA_MODEL

llm_model__: ChatOllama = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0.2,
    tools=[get_downloads_directory],  # tool in stile OpenAI
)


llm_model = ChatOpenAI(
    model=OLLAMA_MODEL,
    base_url="http://localhost:11434/v1",
    api_key="not-needed",
    tools=[get_downloads_directory],  # tool in stile OpenAI
    tool_choice="auto",
)

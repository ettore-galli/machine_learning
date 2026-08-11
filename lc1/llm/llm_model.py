from langchain_ollama import ChatOllama

from llm.base import OLLAMA_MODEL

llm_model: ChatOllama = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0.2,
)

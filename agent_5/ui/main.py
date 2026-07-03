from typing import Any, cast

from ai_agent.base import build_agent_input, extract_response_message
from ai_agent.agent_openai import initialize_agent
from langchain.agents.middleware.types import InputAgentState
from langchain_core.messages import BaseMessage
from langgraph.graph.state import CompiledStateGraph


def perform_model_interaction(agent: CompiledStateGraph, user_prompt: str) -> None:
    agent_input: InputAgentState = build_agent_input(initial_user_prompt=user_prompt)
    response = agent.invoke(agent_input)
    print(extract_response_message(response))


def get_token_content(
    token: str | BaseMessage,
) -> str | list[str | dict[Any, Any]] | None:
    if isinstance(token, str) and token:
        return token
    return cast(BaseMessage, token).content if token is not None else None


def perform_streamed_model_interaction(
    agent: CompiledStateGraph, user_prompt: str
) -> None:
    agent_input: InputAgentState = build_agent_input(initial_user_prompt=user_prompt)

    for token, _ in agent.stream(agent_input, stream_mode="messages"):
        token_content = get_token_content(token=token)

        if token_content:
            print(token_content, end="", flush=True)
    print("\n~°~\n")


def main():
    print("Mini-Agent Locale (LangGraph + Llama.cpp)")
    print("Scrivi 'exit' per uscire.\n")

    agent: CompiledStateGraph = initialize_agent()

    while True:
        user_prompt = input("<prompt>: ")

        if user_prompt == "exit":
            break

        perform_model_interaction(agent=agent, user_prompt=user_prompt)


if __name__ == "__main__":
    main()

from ai_agent.base import build_agent_input, extract_response_message
from ai_agent.agent_openai import initialize_agent
from langchain.agents.middleware.types import InputAgentState
from langgraph.graph.state import CompiledStateGraph


def perform_model_interaction(agent: CompiledStateGraph, user_prompt: str) -> None:
    agent_input: InputAgentState = build_agent_input(initial_user_prompt=user_prompt)
    response = agent.invoke(agent_input)
    print(extract_response_message(response))


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

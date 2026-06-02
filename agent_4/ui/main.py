from agent.agent_openai_tooling import agent_executor
from agent.base import get_initial_agent_state


def main():
    print("Mini-Agent Locale (LangGraph + Llama.cpp)")
    print("Scrivi 'exit' per uscire.\n")

    while True:
        user_prompt = input("<prompt>: ")

        if user_prompt == "exit":
            break

        state = get_initial_agent_state(initial_user_prompt=user_prompt)
        state = agent_executor.invoke(state)

        print("Agente:", state["messages"][-1]["content"], "\n")


if __name__ == "__main__":
    main()

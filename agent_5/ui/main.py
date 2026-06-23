from ai_agent.base import build_agent_input, extract_response_message
from ai_agent.agent_openai import initialize_agent


def main():
    print("Mini-Agent Locale (LangGraph + Llama.cpp)")
    print("Scrivi 'exit' per uscire.\n")

    agent = initialize_agent()

    while True:
        user_prompt = input("<prompt>: ")

        if user_prompt == "exit":
            break

        response = agent.invoke(build_agent_input(initial_user_prompt=user_prompt))

        print(extract_response_message(response))


if __name__ == "__main__":
    main()

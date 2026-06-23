from agent.agent_openai import model


def main():
    print("Mini-Agent Locale (LangGraph + Llama.cpp)")
    print("Scrivi 'exit' per uscire.\n")

    while True:
        user_prompt = input("<prompt>: ")

        if user_prompt == "exit":
            break

        response = model.invoke(user_prompt)

        print(response.content)


if __name__ == "__main__":
    main()

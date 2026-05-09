from agent import agent_executor

def main():
    print("Mini-Agent Locale (LangGraph + Llama.cpp)")
    print("Scrivi 'exit' per uscire.\n")

    state = {"messages": []}

    while True:
        user = input("Tu: ")
        if user == "exit":
            break

        state["messages"].append({"role": "user", "content": user})
        state = agent_executor.invoke(state)

        print("Agente:", state["messages"][-1]["content"], "\n")


if __name__ == "__main__":
    main()

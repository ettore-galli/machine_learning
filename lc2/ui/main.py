import uuid

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from llm.llm_model import llm_agent


def create_thread_id() -> str:
    return str(uuid.uuid4())


def create_config() -> RunnableConfig:
    return {"configurable": {"thread_id": create_thread_id()}}


def provide_user_output(content: str, *args, **kwargs) -> None:
    print(content, *args, **kwargs)


def get_user_input(prompt: str = ">>>") -> str:
    provide_user_output(prompt, end=" ")
    return input()


def chat_loop(config: RunnableConfig) -> None:
    exit_words: list[str] = ["exit", "quit"]

    def should_quit(user_input) -> bool:
        return user_input.lower().split() in exit_words

    while not should_quit(user_input := get_user_input()):
        prompt = (
            """Sei un assistente. Se la richiesta lo prevede, usa i tool a disposizione. """
            + user_input
        )
        response = llm_agent.invoke(
            {"messages": [HumanMessage(content=prompt)]}, config=config
        )
        provide_user_output(response["messages"][-1].content)
        provide_user_output("\n")


def main():
    config = create_config()
    chat_loop(config=config)


if __name__ == "__main__":
    main()

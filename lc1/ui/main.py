from llm.llm_model import llm_model
from langchain_core.messages import HumanMessage

def provide_user_output(content: str, *args, **kwargs) -> None:
    print(content, *args, **kwargs)


def get_user_input(prompt: str = ">>>") -> str:
    provide_user_output(prompt, end=" ")
    return input()


def chat_loop() -> None:
    exit_words: list[str] = ["exit", "quit"]

    def should_quit(user_input) -> bool:
        return user_input.lower().split() in exit_words

    while not should_quit(user_input := get_user_input()):
        prompt = (
            """Sei un assistente. Se la richiesta lo prevede, usa i tool a disposizione. """
            + user_input
        )
        response = llm_model.invoke([HumanMessage(prompt)])
        provide_user_output(response.content)
        provide_user_output("\n")


def main():
    chat_loop()


if __name__ == "__main__":
    main()

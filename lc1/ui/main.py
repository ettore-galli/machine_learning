from llm.llm_model import llm_model


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
        for token in llm_model.stream(user_input):
            provide_user_output(str(token.content), end="")
        provide_user_output("\n")


def main():
    chat_loop()


if __name__ == "__main__":
    main()

from llm.llm_model import llm_model


def provide_user_output(content: str) -> None:
    print(content)


def main():
    response = llm_model.invoke("Ciao!")
    provide_user_output(content=response.content)


if __name__ == "__main__":
    main()

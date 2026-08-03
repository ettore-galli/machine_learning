from typing import Any, cast

from ai_agent.agent_proxy import get_model_response

SYSTEM_PROMPT = ""
EXIT_WORD = "/bye"


def perform_model_interaction(user_prompt: str) -> None:
    response = get_model_response(user_prompt=user_prompt, system_prompt=SYSTEM_PROMPT)
    print(response)


def main():

    print(f"Scrivi '{EXIT_WORD}' per uscire.\n")

    while True:
        user_prompt = input("---> ")

        if user_prompt == EXIT_WORD:
            break

        perform_model_interaction(user_prompt=user_prompt)


if __name__ == "__main__":
    main()

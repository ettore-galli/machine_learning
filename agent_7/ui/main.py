from ai_agent.agent_proxy import get_model_response

SYSTEM_PROMPT = """
    Sei un assistente utile. Hai a disposizione i seguenti strumenti:

    1. get_weather(city): per ottenere il meteo.
    2. calculate_sum(a, b): per sommare due numeri.
    3. calculate_product(a, b): per moltiplicare due numeri.
    4. eval_expression(expr): per valutare un'espressione numerica più complessa
    2. get_dir_list(root): per ottenere la lista di directory


    Se l'utente chiede il meteo o una somma, rispondi nel seguente formato:

    ACTION: nome_funzione, parametro1, parametro2, ecc ecc

    Ad esempio

    ACTION: get_weather, "Mulazzano"
    ACTION: calculate_sum, 17, 34

    Se l'utente fa una richiesta diversa, rispondi normalmente.
    """
EXIT_WORD = "/bye"
USER_PROMPT_MARKER: str = "Utente"


def display_response_segment(segment: str) -> None:
    print(segment, end="", flush=True)


def perform_model_interaction(user_prompt: str) -> None:
    _ = get_model_response(
        user_prompt=user_prompt,
        system_prompt=SYSTEM_PROMPT,
        user_prompt_marker=USER_PROMPT_MARKER,
        response_displayer=display_response_segment,
    )


def main():

    print(f"Scrivi '{EXIT_WORD}' per uscire.\n")

    while True:
        user_prompt = input("---> ")

        if user_prompt == EXIT_WORD:
            break

        perform_model_interaction(user_prompt=user_prompt)


if __name__ == "__main__":
    main()

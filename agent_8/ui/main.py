from pathlib import Path

from ai_agent.agent_proxy import get_model_response
from ai_agent.tools import extract_text, scan_folder

CLASSIFY_SYSTEM_PROMPT = """
Sei un classificatore di documenti.
Dato il seguente testo, restituisci una ed una sola parola che rappresenta il tema principale trattato
- UNA SOLA PAROLA
- NON giustificare la scelta, fornisci solo il risultato

Testo:
    """

GENERAL_SYSTEM_PROMPT = """
Sei un assistente generico. Rispondi alle domande in modo preciso e conciso"""

EXIT_WORD = "/bye"

USER_PROMPT_MARKER: str = "Utente"


def display_response_segment(segment: str) -> None:
    print(segment, end="", flush=True)


def empty(*args, **kwargs) -> None:
    _ = args, kwargs


def perform_model_interaction(user_prompt: str) -> None:
    _ = get_model_response(
        user_prompt=user_prompt,
        system_prompt=GENERAL_SYSTEM_PROMPT,
        user_prompt_marker=USER_PROMPT_MARKER,
        response_displayer=display_response_segment,
    )


def classify_file_content(content: str) -> str:
    return get_model_response(
        user_prompt=content,
        system_prompt=CLASSIFY_SYSTEM_PROMPT,
        user_prompt_marker="Testo:",
        response_displayer=empty,
    )


def main():

    print(f"Scrivi '{EXIT_WORD}' per uscire.\n")

    default = "/users/ettoregalli/Downloads"

    while True:
        user_prompt = input(f"Cartella da classificare: [{default}]")

        if user_prompt == EXIT_WORD:
            break

        user_prompt = user_prompt or default

        print(f"Reading files in {user_prompt}")

        for item in scan_folder(user_prompt):
            file = item["path"]
            try:
                content = extract_text(path=Path(file))
                if content:
                    classification = classify_file_content(content[:1000])
                    print(f"----- {file}, {classification}")
            except UnicodeDecodeError as unicode_error:
                print(f"----- ERR: {file}, {unicode_error}")
            except Exception as other:  # noqa: BLE001
                print(f"----- ERR: {file}, {other}")


if __name__ == "__main__":
    main()

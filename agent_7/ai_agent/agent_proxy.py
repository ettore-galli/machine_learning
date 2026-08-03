from typing import Callable

import ollama


def display_response_segment(segment: str) -> None:
    print(segment, end="", flush=True)


def get_model_response(
    user_prompt: str,
    system_prompt: str,
    user_prompt_marker: str = "Utente",
    response_displayer: Callable[[str], None] = display_response_segment,
) -> str:
    response_segments = []
    for chunk in ollama.generate(
        model="Qwen2.5-Coder:7b",
        prompt=f"{system_prompt}\n{user_prompt_marker}: {user_prompt}",
        stream=True,
    ):
        segment = chunk["response"]

        response_segments.append(segment)

        response_displayer(segment=segment)

    response_displayer(segment="\n")

    return "".join(response_segments)

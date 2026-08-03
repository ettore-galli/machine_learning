import ollama


def get_model_response(
    user_prompt: str, system_prompt: str, user_prompt_marker: str = "Utente"
) -> str:

    response = ollama.generate(
        model="Qwen2.5-Coder:7b",
        prompt=f"{system_prompt}\n{user_prompt_marker}: {user_prompt}",
    )
    return response["response"]

import requests


class OllamaClient:
    def __init__(self, model: str, url: str):
        self.model = model
        self.url = url

    def chat(self, system_prompt, user_prompt, temperature=None, max_tokens=None):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        payload = {"model": self.model, "messages": messages, "stream": False}

        # override opzionali
        if temperature is not None:
            payload["options"] = payload.get("options", {})
            payload["options"]["temperature"] = temperature

        if max_tokens is not None:
            payload["options"] = payload.get("options", {})
            payload["options"]["num_predict"] = max_tokens

        r = requests.post(self.url, json=payload)
        return r.json()["message"]["content"]


"""
client = OllamaClient()

response = client.chat(
    system_prompt="You are a helpful assistant.",
    user_prompt="Spiegami la differenza tra runtime.",
    temperature=0.7,
    max_tokens=512,
)

print(response)
"""

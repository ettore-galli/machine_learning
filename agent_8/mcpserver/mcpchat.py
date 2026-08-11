import json
import requests
from mcpserver.server import mcp

MODEL = "qwen2.5-coder:7b"


def debug_history(history):
    print("-" * 80)
    for item in history:
        print(f"* {item["role"].strip().upper()}:{item["content"]}")



def call_ollama(history):
    """Invia la conversazione a Ollama e ottiene la risposta."""
    payload = {"model": MODEL, "messages": history, "stream": False}
    r = requests.post("http://localhost:11434/api/chat", json=payload)
    return r.json()


def extract_tool_call(msg):
    """Estrae una tool call dallo stile OpenAI/Ollama."""
    try:
        content = msg["message"]["content"]
        data = json.loads(content)
        return data.get("tool_call")
    except:
        return None


# history = [{"role": "user", "content": "Mostra all'utente il risultato di 3*(2*56)-4"}]
history = [{"role": "user", "content": "calcola il numero di ettore usado i tool a disposizione "
" per 5"}]


while True:
    # 1. Chiedi a Ollama cosa fare
    response = call_ollama(history)
    debug_history(history)

    msg = response["message"]

    # 2. Controlla se c’è una tool call
    tool_call = extract_tool_call(response)

    if tool_call:
        # 3. Esegui il tool via MCP
        print(f"USING TOOL {tool_call["arguments"]}")
        result = mcp.call_tool(tool_call["name"], tool_call["arguments"])

        # 4. Rimanda il risultato all’LLM
        history.append({"role": "tool", "content": json.dumps(result)})

        continue

    # 5. Nessuna tool call → risposta finale
    history.append(msg)
    break

print("Risposta finale:", msg["content"])

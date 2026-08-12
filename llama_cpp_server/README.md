# LLAMA CPP SERVER

## Download

Releases home:

<https://github.com/ggml-org/llama.cpp/releases>

Direct link

<https://github.com/ggml-org/llama.cpp/releases/download/b10373/llama-b10373-bin-macos-arm64.tar.gz>

## Setup

```shell
https://github.com/ggml-org/llama.cpp/tree/master

mkdir build
cd build


cmake -DGGML_METAL=ON ..
```

## SETUP AI

Il percorso di setup **llama.cpp server** è molto più semplice di quanto sembri, ma ci sono alcuni passaggi critici che vanno fatti nell’ordine giusto per ottenere **un backend realmente OpenAI‑compatible**, così puoi usare:

- `ChatOpenAI`
- `tools=[...]`
- `tool_choice="auto"`
- JSON mode
- streaming dei tool‑calls

Qui sotto ti metto **il percorso completo**, ordinato e senza ambiguità, così puoi integrarlo nel tuo progetto LC1.

---

## 🧩 1. Scarica llama.cpp (versione con server OpenAI)

Serve la build che include:

- `llama-server`
- endpoint `/v1/chat/completions`

Il repository è questo:

- **llama.cpp**

Clona:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
```

---

## 🧩 2. Compila con supporto server

Su macOS (M2/M3):

```bash
make server
```

Questo genera:

```
./server
```

---

## 🧩 3. Avvia il server in modalità OpenAI‑compatible

Esempio:

```bash
./server \
  --model /percorso/al/modello/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key not-needed \
  --chatml
```

Parametri importanti:

- `--chatml` → abilita il formato OpenAI
- `--api-key` → richiesto da ChatOpenAI (può essere qualsiasi stringa)
- `--model` → punta al tuo GGUF

---

## 🧩 4. Test rapido dell’API OpenAI

Da terminale:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer not-needed" \
  -d '{
    "model": "local-model",
    "messages": [{"role": "user", "content": "Ciao"}]
  }'
```

Se risponde con JSON stile OpenAI → tutto ok.

---

## 🧩 5. Usa ChatOpenAI con tool‑calling automatico

Ora puoi usare **ChatOpenAI** senza errori Pydantic:

```python
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage

@tool
def addizione(x: int, y: int) -> int:
    """Somma due numeri."""
    return x + y

llm = ChatOpenAI(
    model="local-model",
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    tools=[addizione],
    tool_choice="auto"
)

response = llm.invoke([
    HumanMessage(content="Somma 2 e 3 usando il tool addizione.")
])

print(response)
```

Con llama.cpp server:

- `tools` viene accettato
- `tool_choice="auto"` funziona
- il modello produce `tool_calls`
- LangChain esegue il tool
- tutto senza hack

---

## 🧩 6. Modelli consigliati per tool‑calling

I tuoi modelli che funzionano meglio:

- **Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf** ⭐ migliore
- **Qwen2.5-Coder-7B.Q4_K_M.gguf**
- **Qwen2.5-Coder-14B-Q5_K_M.gguf**

Funzionano anche:

- **mistral-7b-Q4_K_M.gguf**

Meno precisi:

- **qwen2.5-1.5b-instruct-q4_k_m.gguf**
- **qwen2.5-3b-instruct-q8_0.gguf**

---

## 🧩 7. Integrazione nel tuo progetto LC1

Il tuo LC1 deve:

1. creare `llm_model = ChatOpenAI(...)`
2. NON usare ChatOllama
3. NON usare Ollama come backend
4. NON passare tools a Ollama
5. usare solo llama.cpp server

---

## 🔥 Sintesi finale

### Se vuoi tool‑calling automatico stile OpenAI

- backend: **llama.cpp server**
- modello: **Llama‑3.1‑8B‑Instruct**
- client: **ChatOpenAI**
- tools: **funzionano nativamente**

### Se usi Ollama

- niente tool‑calling automatico
- solo tool‑calling manuale

---

## Vuoi che ti preparo

- script completo di avvio server + test + client Python
- template LC1 con tool‑calling automatico
- configurazione multi‑modello con router

Dimmi quale preferisci.

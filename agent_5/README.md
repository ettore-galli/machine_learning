# AGENT 5

## Corso LangChain

<https://academy.langchain.com/courses/take/foundation-introduction-to-langchain-python>

## Download modello (una tantum)

```sh
curl -L https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf -o /Volumes/DOCKER/huggingface/downloaded/qwen2.5-1.5b-instruct-q4_k_m.gguf

curl -L https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q8_0.gguf -o /Volumes/DOCKER/huggingface/downloaded/qwen2.5-3b-instruct-q8_0.gguf
```

## Setup uv (una tantum)

```shell
export UV_INSTALL_DIR="$(pwd)/tools/uv"
curl -LsSf https://astral.sh/uv/install.sh | sh

```

```shell
export PATH="$(pwd)/tools/uv:$PATH"
```

## Init uv (una tantum)

```shell
uv init --no-workspace
```

```shell

uv add "langchain>=0.3.0"
uv add "langchain-community>=0.3.0"
uv add "langgraph>=0.2.59"
uv add "llama-cpp-python>=0.2.90"
uv add "llama-cpp-python[server]"
uv add "python-dotenv>=1.0.1"

uv add --dev ruff
uv add --dev black
uv add --dev pyright
```

## Llama cpp server

Start:

```shell
./llama_cpp_server.sh
```

Chiamata:

```shell
curl http://localhost:9876/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "local-model",
        "messages": [{"role": "user", "content": "Rispondi unicamente si o no: 3+7 è un calcolo?"}]
      }'
```

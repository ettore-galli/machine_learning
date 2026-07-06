# AGENT 5

## Corso LangChain

<https://academy.langchain.com/courses/take/foundation-introduction-to-langchain-python>

## Download modello (una tantum)

```sh
curl -L https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf -o /Volumes/DOCKER/huggingface/downloaded/qwen2.5-1.5b-instruct-q4_k_m.gguf

curl -L https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q8_0.gguf -o /Volumes/DOCKER/huggingface/downloaded/qwen2.5-3b-instruct-q8_0.gguf


curl -L https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/mistral-7b-instruct-v0.3.Q4_K_M.gguf -o /Volumes/DOCKER/huggingface/downloaded/mistral-7b-instruct-v0.3.Q4_K_M.gguf 

```

## Setup uv (una tantum)

```shell

export UV_INSTALL_DIR="$(pwd)/tools/uv"
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Oppure:

```shell
mkdir tools

curl -LsSf https://astral.sh/uv/install.sh | \
  env UV_INSTALL_DIR="$(pwd)/tools/uv" \
      UV_NO_MODIFY_PATH=1 sh

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

## VLLM Server setup

Model

```shell
hf download Qwen/Qwen2.5-3B-Instruct --local-dir /Volumes/DOCKER/hf_models/qwen2.5-3b
hf download mistralai/Mistral-7B-Instruct-v0.3 --local-dir /Volumes/DOCKER/hf_models/mistral7b

python convert-hf-to-gguf.py \
  /Volumes/DOCKER/hf_models/mistral7b \
  --out /Volumes/DOCKER/huggingface/downloaded/mistral7b.gguf

```

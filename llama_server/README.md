# SETUP LLAMA SERVER

## Download (once)

```sh
https://ollama.com/download/Ollama-darwin.zip
```

## Extract app (once)

Ollama.app

## Create Modelfile (once)

e.g.

```Modelfile
FROM /Volumes/DOCKER/huggingface/downloaded/qwen2.5-3b-instruct-q8_0.gguf
```

## Create (once)

```sh
llama_server/ollama-create.sh
```

## Startup

```sh
# Start
ollama-start.sh

# Check
curl http://localhost:11434/api/version
```

## Avvio modello

```sh
# Start
ollama-run.sh

# Check
curl http://localhost:11434/api/version
```

## Usa modello

```shell

curl http://localhost:11434/api/generate -d '{
  "model": "qwen-custom-1",
  "prompt": "Why is the sky blue?"
}'

```

## Una tantum: spostamento su SSD del modello locale

```shell

# Standerd ollama:
#   ~/.ollama/models
# Link percorso
export OLLAMA_STANDARD_DIR="$HOME/.ollama/models"
export OLLAMA_EXTERNAL_DIR="/Volumes/DOCKER/ollama-models"

mv "${OLLAMA_STANDARD_DIR}" "${OLLAMA_EXTERNAL_DIR}"

ln -s "${OLLAMA_EXTERNAL_DIR}" "${OLLAMA_STANDARD_DIR}"

```

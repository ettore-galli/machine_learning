# SETUP OLLAMA LOCAL

## Download and place ollama locally

```txt
https://ollama.com/download/mac
```

## Create an environment

@see env.sh

## One time setup script

@see setup.sh

## Daily start script

@see serve.sh
@see cli.sh

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

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

## Installazione

```shell


sudo /Applications/Ollama.app/Contents/MacOS/Ollama install


ollama --version
ollama ps
```

## Gestione daemon

```shell
# 1) Disabilita il servizio launchd
#    Dopo l’installazione, esegui:

sudo launchctl disable system/com.ollama.ollama

# 1) (Opzionale) Ferma il daemon se è già in esecuzione
sudo launchctl stop system/com.ollama.ollama

#2) Avvia Ollama solo quando ti serve

ollama serve

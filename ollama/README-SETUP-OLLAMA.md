# SETUP OLLAMA

## [Once] Install via official installer

```txt
https://ollama.com/download/mac
```

* "Download for Mac OS"
* Copy into Applications

Then perform:

```shell
sudo /Applications/Ollama.app/Contents/MacOS/Ollama install

ollama --version
ollama ps
```

## [Once] Create an environment

@see env.sh

## [Once] One time setup script

@see setup.sh

## [Daily] Daily start script

@see serve.sh
@see cli.sh

## [Once] move local models onto external SSD

```shell

# Standard ollama model path:
#   ~/.ollama/models
# Path link
export OLLAMA_STANDARD_DIR="$HOME/.ollama/models"
export OLLAMA_EXTERNAL_DIR="/Volumes/DOCKER/ollama-models"

mv "${OLLAMA_STANDARD_DIR}" "${OLLAMA_EXTERNAL_DIR}"

ln -s "${OLLAMA_EXTERNAL_DIR}" "${OLLAMA_STANDARD_DIR}"

```

## Manage ollama daemon

i.e. disable ollama daemon for manual start via ollama serve

```shell
sudo launchctl disable system/com.ollama.ollama
sudo launchctl stop system/com.ollama.ollama
```

## Start ollama daemon

```shell
ollama serve
```

## [Once] Obtain model

Example

```shell
export OLLAMA_MODELS="/Maybe/your/external/drive"

ollama pull qwen2.5-coder:7b
```

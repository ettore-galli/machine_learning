# OPENCODE

## Reference

```txt
https://opencode.ai/docs/
https://ollama.com/library/llama3.1:8b-instruct-q5_K_M
https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF?utm_source=copilot.com
```

## Setup opencode

```shell
curl -fsSL https://opencode.ai/install | bash
```

~/.config/opencode/opencode.json

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama (local)",
      "options": {
        "baseURL": "http://localhost:11434/v1"
      },
      "models": {
        "Qwen2.5-Coder": {
          "name": "Qwen2.5-Coder:latest",
          "tools": true
        }
      }
    }
  },
  "model": "ollama/Qwen2.5-Coder"
}
```

## Setup ollama

```shell
curl -fsSL https://ollama.com/install.sh | sh

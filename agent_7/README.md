# AGENT 7

ESEMPIO INTERESSANTE DI CHAT CHE FUNZIONA DECENTEMENTE
Ollama + qwen2.5-coder:7b

## Setup

## Init uv (una tantum)

```shell
uv init --no-workspace
```

```shell

uv add "ollama==0.6.2"

uv add --dev ruff
uv add --dev black
uv add --dev pyright
```

## Ollama

Start:

```shell
ollama serve
```

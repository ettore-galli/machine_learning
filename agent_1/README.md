# AGENT 1

Il primo, non ho idea di cosa faccia, vediamo strada facendo

Il prossimo sarà molto probabilmente agent-2

## Setup uv

```shell
export UV_INSTALL_DIR="$(pwd)/tools/uv"
curl -LsSf https://astral.sh/uv/install.sh | sh

```

```shell
export PATH="$(pwd)/tools/uv:$PATH"
```

## Init uv

```shell
uv init --no-workspace
```

```shell
uv add langchain
uv add langchain-huggingface
uv add transformers
uv add duckduckgo-search
uv add python-dotenv
uv add beautifulsoup4

uv add --dev ruff
uv add --dev black
uv add --dev pyright
```

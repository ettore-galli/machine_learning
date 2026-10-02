# SIMPLE NN


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
uv add pytorch
uv add matplotlib

uv add --dev ruff
uv add --dev black
uv add --dev pyright
```


add-uv-path:
	@echo 'export PATH=$$(pwd)/tools/uv:$$PATH'

setup-lint:
	uv tool install --active ruff
	uv tool install --active pyright
	uv tool install --active black

test:
	pytest -s -vvv tests/unit/

lint:
	uv run --active black .
	uv run --active ruff check .
	uv run --active pyright .

ruff-fix:
	uv run --active ruff check . --fix

sync:
	uv sync --active

install-python:
	uv python install 3.12
	uv python update-shell

create:
	uv venv --clear --python python3.12 .venv 

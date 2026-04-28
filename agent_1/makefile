
add-uv-path:
	@echo 'export PATH=$$(pwd)/tools/uv:$$PATH'

setup-lint:
	uv tool install ruff
	uv tool install pyright
	uv tool install black


lint:
	uv run black .
	uv run ruff check .
	uv run black .

ruff-fix:
	uv run ruff check . --fix

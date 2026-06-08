packages=fastai_work
 

lint:
	uv run black $(packages)
	uv run ruff check $(packages)
	uv run pyright $(packages)
packages=fastai_work
 
lint:
	black $(packages)
	ruff check $(packages)
	mypy $(packages)

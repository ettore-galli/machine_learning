
PACKAGE=perceptron

test: 
	@echo "===== TESTING ${PACKAGE} ====="
	pytest tests/

lint:
	@echo "===== LINTING ${PACKAGE} ====="
	flake8 ${PACKAGE}
	black -t py311 ${PACKAGE}
	pylint ${PACKAGE}
	mypy ${PACKAGE}

all: lint test

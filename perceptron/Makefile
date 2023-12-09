
PACKAGE=perceptron

lint:
	@echo "===== LINTING ${PACKAGE} ====="
	flake8 ${PACKAGE}
	black -t py311 ${PACKAGE}
	pylint ${PACKAGE}
	mypy ${PACKAGE}


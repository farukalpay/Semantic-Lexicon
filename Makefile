.PHONY: fmt lint type test docs all

fmt:
	black src tests

lint:
	ruff check src tests

type:
	mypy src

test:
	pytest

docs:
	mkdocs build

all: fmt lint type test docs

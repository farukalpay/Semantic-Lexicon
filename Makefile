.PHONY: format lint type test docs

format:
	black src tests

lint:
	ruff check src tests
	ruff format --check src tests

type:
	mypy src

unit:
	pytest

test: lint type unit

docs:
	mkdocs build

# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

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

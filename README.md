# Semantic Lexicon

Semantic Lexicon is a NumPy-first research toolkit that has been refactored
into a production-quality Python package. The project now ships with a modular
package layout, deterministic configuration handling, an end-to-end CLI, and
first-class automation for testing, documentation, and packaging.

## Key Capabilities

- **Modular architecture** – dedicated submodules for embeddings, intents,
  knowledge graph management, persona vectors, and persona-aware generation.
- **Deterministic workflows** – configuration dataclasses serialise to YAML/JSON
  and a reproducible seeding strategy keeps automated tests stable.
- **Typer-based CLI** – prepare corpora, run lightweight NumPy training loops,
  execute diagnostics, and generate persona-conditioned responses from the
  command line.
- **Automation ready** – Makefile targets, pytest suite, linting/typing
  configuration, and MkDocs documentation prepare the repository for CI/CD.

## Installation

```bash
python -m pip install .[dev,docs]
```

Runtime dependencies are intentionally minimal (`numpy`, `typer`, `PyYAML`). The
`dev` and `docs` extras install tooling for tests, linting, typing, and
documentation builds.

## Quick Start

```bash
# bootstrap configuration
semantic-lexicon init-config

# prepare the bundled sample corpus
semantic-lexicon prepare src/semantic_lexicon/data/sample_corpus.json --output processed.json

# train and persist artefacts
semantic-lexicon train processed.json --workspace .semantic_lexicon

# run diagnostics and export a JSON report
semantic-lexicon diagnostics --output diagnostics.json

# generate persona-aware responses
semantic-lexicon generate "Explain neural networks" --persona storyteller
```

Programmatic usage mirrors the CLI:

```python
from semantic_lexicon import NeuralSemanticModel, SemanticModelConfig
from semantic_lexicon.training import CorpusProcessor, Trainer
from semantic_lexicon.data import load_sample_corpus

config = SemanticModelConfig()
model = NeuralSemanticModel(config=config)

processor = CorpusProcessor()
entries = processor.prepare_corpus(load_sample_corpus())

trainer = Trainer(model, config=config.trainer)
trainer.train(entries)
model.save(config.workspace)

response = model.generate_response("Explain gradient descent", persona="tutor")
print(response)
```

## Repository Layout

```
├── docs/                    # MkDocs documentation
├── mkdocs.yml               # Documentation build configuration
├── pyproject.toml           # Packaging metadata and tooling config
├── src/semantic_lexicon/    # Library source tree
│   ├── cli.py               # Typer-based CLI entry points
│   ├── config.py            # Dataclasses and config utilities
│   ├── embeddings.py        # GloVe-style embeddings and persistence
│   ├── generator.py         # Persona-aware generator
│   ├── intent.py            # Intent classifier implementation
│   ├── knowledge.py         # Knowledge network helpers
│   ├── model.py             # NeuralSemanticModel façade
│   ├── persona.py           # Persona registry
│   ├── pipelines/           # High-level automation flows
│   ├── training.py          # Corpus processing and trainer
│   └── utils/               # Shared helpers (text, IO, randomness)
├── tests/                   # Pytest suite with fixtures and CLI coverage
└── Makefile                 # Automation targets (fmt, lint, type, test, docs)
```

## Automation & Quality

- `make fmt` – format code with Black.
- `make lint` – run Ruff lint checks.
- `make type` – execute MyPy static typing.
- `make test` – run pytest-based unit tests.
- `make docs` – build MkDocs documentation.

These commands can be orchestrated in CI (e.g., GitHub Actions) to enforce code
quality for pull requests.

## Diagnostics

`semantic_lexicon.diagnostics` exposes a structured `DiagnosticsSuite` that can
be used from Python or the CLI. The suite reports embedding norms, intent
predictions, persona statistics, and generation previews. When optional
dependencies like `rich` or `pandas` are installed the diagnostics output can be
rendered as tables or DataFrames.

## Contributing

Contributions are welcome! Please run the formatting, linting, typing, testing,
and documentation commands before submitting pull requests. For substantial
changes, update the docs under `docs/` and ensure the README highlights the new
capabilities.

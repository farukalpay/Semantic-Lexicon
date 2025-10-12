# Semantic Lexicon

Semantic Lexicon is a NumPy-first research toolkit that demonstrates persona-aware semantic modelling. The project packages a compact neural stack consisting of intent understanding, a light-weight knowledge network, persona management, and text generation into an automated Python library and CLI.

## Features

- **Modular architecture** – dedicated submodules for embeddings, intent classification, knowledge graphs, persona handling, and persona-aware generation.
- **Deterministic NumPy training loops** – simple yet reproducible optimisation routines for intents and knowledge edges.
- **Automated workflows** – Typer-powered CLI (`semantic-lexicon`) for corpus preparation, training, diagnostics, and generation.
- **Extensible configuration** – YAML/JSON configuration loading with dataclass-backed defaults.
- **Diagnostics** – structured reports covering embeddings, intents, knowledge neighbours, personas, and generation previews.
- **Docs & tests** – MkDocs documentation, pytest-based regression tests, and CI-ready tooling (black, ruff, mypy).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install .[dev,docs]
```

To install the core package only:

```bash
pip install .
```

## Project Layout

```
src/semantic_lexicon/
├── cli.py                # Typer CLI entry point
├── config.py             # Dataclass-backed configuration helpers
├── embeddings.py         # GloVe-style embeddings and persistence
├── intent.py             # NumPy multinomial logistic regression intents
├── knowledge.py          # Simple relation network with gradient updates
├── persona.py            # Persona profiles and blending logic
├── generator.py          # Persona-aware response generation
├── model.py              # Orchestration façade
├── training.py           # Training pipeline and diagnostics integration
├── diagnostics.py        # Structured diagnostics reports
├── utils/                # Tokenisation, seeding, and I/O helpers
└── data/                 # Sample intent & knowledge datasets for tests
```

## Quick Start

1. **Prepare the corpus** (optional if using bundled sample data):

   ```bash
   semantic-lexicon prepare --intent src/semantic_lexicon/data/intent.jsonl --knowledge src/semantic_lexicon/data/knowledge.jsonl --workspace artifacts
   ```

2. **Train the model** (uses processed datasets in `artifacts/`):

   ```bash
   semantic-lexicon train --workspace artifacts
   ```

   The CLI saves embeddings, intent weights, and knowledge matrices to the workspace directory.

3. **Run diagnostics**:

   ```bash
   semantic-lexicon diagnostics --workspace artifacts --output diagnostics.json
   ```

   The command prints a JSON summary to stdout and optionally writes the report to disk.

4. **Generate responses**:

   ```bash
   semantic-lexicon generate "Explain neural networks" --workspace artifacts --persona tutor
   ```

## Configuration

Semantic Lexicon reads configuration files in YAML or JSON using the `SemanticModelConfig` dataclass. Example `config.yaml`:

```yaml
embeddings:
  dimension: 50
  max_words: 5000
intent:
  learning_rate: 0.2
  epochs: 5
knowledge:
  max_relations: 4
persona:
  default_persona: tutor
generator:
  temperature: 0.7
```

Load the configuration via CLI (`semantic-lexicon train --config config.yaml`) or programmatically:

```python
from semantic_lexicon import NeuralSemanticModel, load_config

config = load_config("config.yaml")
model = NeuralSemanticModel(config)
```

## Training API

```python
from semantic_lexicon import NeuralSemanticModel, SemanticModelConfig
from semantic_lexicon.training import Trainer, TrainerConfig

config = SemanticModelConfig()
model = NeuralSemanticModel(config)
trainer = Trainer(model, TrainerConfig())
trainer.train()
response = model.generate("How to learn python?", persona="tutor")
print(response.response)
```

## Diagnostics Programmatically

```python
from semantic_lexicon.model import NeuralSemanticModel
from semantic_lexicon.training import Trainer, TrainerConfig

model = NeuralSemanticModel()
trainer = Trainer(model, TrainerConfig())
trainer.train()
report = trainer.run_diagnostics()
print(report.to_dict())
```

## Development Workflow

- **Format & lint**: `ruff check .` and `black .`
- **Type check**: `mypy src`
- **Tests**: `pytest`
- **Docs**: `mkdocs serve`

A `Makefile` (or CI workflow) can orchestrate the tasks:

```bash
make lint
make test
make docs
```

## Contributing

1. Fork the repository and create a feature branch.
2. Install development dependencies: `pip install .[dev]`.
3. Run `make test` to ensure linting, typing, and tests pass.
4. Submit a pull request with detailed notes on new features or fixes.

## License

Semantic Lexicon is released under the MIT License. See [LICENSE](LICENSE) for details.

# Semantic Lexicon

Semantic Lexicon is a research-grade Python library that bundles together the
components of the “neural semantic model” developed in this repository.  The
package focuses on learned embeddings, intent understanding, a lightweight
knowledge graph, and persona-conditioned text generation – all implemented in
pure NumPy so the system can run without a deep-learning framework.

## Features

- **NeuralSemanticModel** – unified façade that wires embeddings, intent
  classification, knowledge retrieval, persona conditioning, and generation.
- **Training utilities** – helpers for preparing corpora, seeding the knowledge
  graph, and running illustrative optimisation loops.
- **Diagnostics suite** – structured smoke tests that mirror the historical
  command-line script and surface key signals for embeddings, intents, and
  generation behaviour.
- **Zero external services** – no runtime dependency on Wikipedia or other
  network calls; behaviour is derived from embeddings and optional training
  data.

## Installation

Semantic Lexicon uses a `src/` layout and standard Python packaging metadata in
`pyproject.toml`.  Install it from a local checkout with:

```bash
python -m pip install .
```

The only runtime dependency is `numpy>=1.23`.  The library targets Python 3.9+
(and runs happily on newer interpreters).

## Quick Start

```python
from semantic_lexicon import NeuralSemanticModel

model = NeuralSemanticModel()
response = model.generate_response(
    query="What is machine learning?",
    persona="tutor",
    max_length=60,
    temperature=0.7,
)
print(response)
```

Retrieve high-level signals about how the model is behaving:

```python
from semantic_lexicon import run_all_diagnostics

result = run_all_diagnostics(stream=None)
print(result.to_dict())
```

## Diagnostics

The former `test_neural_model.py` script now lives behind the
`semantic_lexicon.diagnostics` module.  The public helpers are:

- `DiagnosticsSuite` – orchestrates the probes and exposes structured results.
- `run_all_diagnostics()` – convenience function that optionally prints a human
  readable report when a `stream` (like `sys.stdout`) is provided.

Example command-line usage:

```python
from semantic_lexicon import run_all_diagnostics
import sys

run_all_diagnostics(stream=sys.stdout)
```

This prints a concise summary including embedding norms, intent predictions,
knowledge retrieval samples, persona vector statistics, and generation
previews.

## Training Helpers

The `semantic_lexicon.training` module provides building blocks for preparing
custom corpora and populating the model:

```python
from semantic_lexicon import NeuralSemanticModel
from semantic_lexicon.training import CorpusProcessor, NeuralTrainer

raw_entries = [
    {"title": "Neural Networks", "summary": "A model inspired by biology."},
    {"title": "Gradient Descent", "summary": "Optimisation procedure."},
]

processor = CorpusProcessor()
processed = processor.prepare_corpus(raw_entries)

model = NeuralSemanticModel()
trainer = NeuralTrainer(model)
training_pairs = trainer.create_training_pairs(processed)
trainer.train_intent_classifier(training_pairs, epochs=3)
trainer.train_knowledge_network(processed)
```

Key classes:

- `CorpusProcessor` – normalises raw records, tokenises text, and extracts
  concept hints for the knowledge graph.
- `NeuralTrainer` – bootstraps the intent classifier, knowledge graph, and
  generator using lightweight illustrative loops.
- `LexiconMigrator` – wraps the neural generator so teams can compare new
  outputs against legacy systems.

All utilities are intentionally framework-agnostic; integrate them into your own
pipelines or replace them with domain-specific variants.

## Architecture Overview

`NeuralSemanticModel` exposes the following subsystems:

1. **Embeddings** – loads or synthesises GloVe-style vectors with consistent
   handling for out-of-vocabulary tokens.
2. **IntentClassifier** – BiLSTM-inspired scaffolding that scores definition,
   comparison, how-to, benefit, identity, and general intents.
3. **KnowledgeNetwork** – attention-weighted concept graph used to retrieve
   supporting facts.
4. **NeuralGenerator** – persona-aware decoder featuring beam search and copy
   heuristics (implemented with NumPy tensors).
5. **PersonalityModule** – transforms base representations into persona-specific
   styles through learned matrices.

The architecture is intentionally modular so researchers can swap out components
or plug in trained weights.

## Repository Layout

```
├── pyproject.toml          # Packaging metadata
├── src/
│   └── semantic_lexicon/
│       ├── __init__.py     # Public API exports
│       ├── model.py        # Core neural architecture
│       ├── diagnostics.py  # Structured diagnostic harness
│       └── training.py     # Corpus and migration helpers
└── Archive/                # Legacy materials retained for reference
```

## Contributing

Issues and pull requests are welcome.  Please ensure new functionality includes
pertinent documentation updates and, where possible, extend the diagnostics or
add automated tests to showcase the behaviour.

## License

Semantic Lexicon is released under the MIT License.  See `LICENSE` for the full
text.

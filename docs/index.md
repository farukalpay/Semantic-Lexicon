# Semantic Lexicon

Welcome to the automated Semantic Lexicon documentation. This site outlines the
modern package layout, automation workflows, and extensibility guidelines for
the NumPy-based semantic language model.

## Quick Start

```bash
pip install semantic-lexicon[dev]
semantic-lexicon init-config
semantic-lexicon prepare src/semantic_lexicon/data/sample_corpus.json
semantic-lexicon train processed_corpus.json
semantic-lexicon diagnostics --output diagnostics.json
```

## Package Highlights

- Modular architecture with dedicated subpackages for embeddings, intents,
  knowledge, personas, and generation.
- Reproducible training with deterministic seeding and serialisable configs.
- Typer-based CLI exposing preparation, training, diagnostics, and generation.
- MkDocs documentation, pytest suite, and GitHub Actions-ready automation.

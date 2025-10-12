# User Guide

This guide highlights the automated workflows available through the Semantic Lexicon CLI.

## Preparing Data

```bash
semantic-lexicon prepare \
  --intent src/semantic_lexicon/data/intent.jsonl \
  --knowledge src/semantic_lexicon/data/knowledge.jsonl \
  --workspace artifacts
```

The command normalises text, writes JSONL files, and stores them in the workspace. Provide your own JSON or JSONL files with keys `text`/`intent` and `head`/`relation`/`tail` to integrate with the trainer.

## Training

```bash
semantic-lexicon train --workspace artifacts
```

Training loads processed datasets, fits the intent classifier and knowledge network, and persists artifacts (`embeddings.json`, `intent.json`, `knowledge.json`).

### Configuration Overrides

Use YAML/JSON configuration files to tweak dimensions or learning hyperparameters:

```bash
semantic-lexicon train --config config.yaml --workspace artifacts
```

## Diagnostics

```bash
semantic-lexicon diagnostics --workspace artifacts --output diagnostics.json
```

Diagnostics executes the `DiagnosticsSuite` and prints structured JSON to stdout. The optional `--output` flag writes the report to disk for integration with dashboards.

## Generation

```bash
semantic-lexicon generate "Share an approach to study machine learning" --workspace artifacts --persona researcher
```

The generator blends prompt context with persona embeddings and knowledge neighbours to produce a short textual response.

## Automation Pipelines

Use the programmatic pipeline to orchestrate training in Python:

```python
from semantic_lexicon.pipelines import prepare_and_train

model = prepare_and_train(workspace="artifacts")
print(model.generate("Explain AI", persona="tutor").response)
```

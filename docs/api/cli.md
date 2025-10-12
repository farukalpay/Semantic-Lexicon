# CLI Reference

Semantic Lexicon exposes a Typer-based CLI installed as `semantic-lexicon`.

## Commands

### `prepare`

```
semantic-lexicon prepare --intent <INTENT.jsonl> --knowledge <KNOWLEDGE.jsonl> --workspace <DIR>
```

Normalises raw JSON/JSONL datasets and stores processed JSONL files in the workspace.

### `train`

```
semantic-lexicon train --workspace <DIR> [--config CONFIG.yaml]
```

Trains the intent classifier and knowledge network, saving learned artifacts in the workspace.

### `diagnostics`

```
semantic-lexicon diagnostics --workspace <DIR> [--config CONFIG] [--output REPORT.json]
```

Runs the diagnostics suite and optionally writes the structured report to disk.

### `generate`

```
semantic-lexicon generate "<PROMPT>" --workspace <DIR> [--persona PERSONA]
```

Generates a persona-conditioned response using the trained model artifacts.

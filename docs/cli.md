# Command Line Interface

The `semantic-lexicon` command exposes the complete automation workflow.

## Prepare

```bash
semantic-lexicon prepare data.json --output processed.json
```

Reads a JSON or YAML corpus and writes a normalised dataset containing tokenised
records and vocabulary statistics.

## Train

```bash
semantic-lexicon train processed.json --config semantic_lexicon.yaml
```

Runs the deterministic NumPy training loops for the intent classifier and
knowledge network, saving artefacts to the configured workspace.

## Diagnostics

```bash
semantic-lexicon diagnostics --output diagnostics.json
```

Executes the diagnostics suite and optionally writes the results to JSON. When
`rich` is installed, a table-based report is rendered in the terminal.

## Generate

```bash
semantic-lexicon generate "Explain gradient descent" --persona researcher
```

Produces a persona-conditioned response using the orchestrated model.

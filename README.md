# Semantic Lexicon

[![CLI Tests](https://github.com/farukalpay/Semantic-Lexicon/actions/workflows/cli-tests.yml/badge.svg)](https://github.com/semantic-lexicon/Semantic-Lexicon/actions/workflows/cli-tests.yml)

Semantic Lexicon is a NumPy-first research toolkit that demonstrates persona-aware semantic modelling. The project packages a compact neural stack consisting of intent understanding, a light-weight knowledge network, persona management, and text generation into an automated Python library and CLI.

The name reflects the long-standing academic concept of the [semantic lexicon](https://en.wikipedia.org/wiki/Semantic_lexicon); this repository contributes an applied, open implementation that operationalises those ideas for persona-aware experimentation.

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

## Lightweight Q&A Demo

Semantic Lexicon can answer short questions after its bundled model components are trained. The stack is intentionally tiny, so
the phrasing is concise, but the generator now runs a compact optimisation loop that:

1. **Classifies intent** with the logistic-regression intent model.
2. **Builds noun-phrase and collocation candidates** whose adjacent tokens clear an adaptive pointwise mutual information (PMI)
   threshold, keeping multi-word ideas intact.
3. **Scores each candidate** via cosine relevance to the blended persona/prompt embedding, tf–idf salience, and a capped PMI
   cohesion bonus.
4. **Selects diverse topics** with Maximum Marginal Relevance (MMR) plus an n-gram overlap penalty so the guidance does not echo
   the question verbatim.
5. **Surfaces knowledge neighbours** starting from the strongest topic for additional context.
6. **Aligns journaling actions** with the detected intent so each topic carries a concise Explore/Practice/Reflect-style cue.

1. Install the project in editable mode:

   ```bash
   pip install -e .
   ```

2. Run a quick script that trains the miniature model and generates answers for a few prompts:

   ```bash
   python - <<'PY'
   from semantic_lexicon import NeuralSemanticModel, SemanticModelConfig
   from semantic_lexicon.training import Trainer, TrainerConfig

   config = SemanticModelConfig()
   model = NeuralSemanticModel(config)
   trainer = Trainer(model, TrainerConfig())
   trainer.train()

   for prompt in [
       "How do I improve my public speaking?",
       "Explain matrix multiplication",
       "What is machine learning?",
       "Tips for staying productive while studying",
       "Clarify the concept of photosynthesis",
       "How can I organize my research presentation effectively?",
       "Define gravitational potential energy",
   ]:
       response = model.generate(prompt, persona="tutor")
       print(f"Prompt: {prompt}\\nResponse: {response.response}\\n")
   PY
   ```

   Sample output after training the bundled data:

   ```
   Prompt: How do I improve my public speaking?
   Response: From a balanced tutor perspective, let's look at How do I improve my public speaking? This ties closely to the 'how_to' intent I detected. Consider journaling about: Public Speaking (Explore), Practice Routine (Practice), Feedback Loops (Reflect). Try to explore Public Speaking, practice Practice Routine, and reflect Feedback Loops.

   Prompt: Explain matrix multiplication
   Response: From a balanced tutor perspective, let's look at Explain matrix multiplication. This ties closely to the 'definition' intent I detected. Consider journaling about: Matrix Multiplication (Define), Dot Products (Explore), Linear Transformations (Compare). Try to define Matrix Multiplication, explore Dot Products, and compare Linear Transformations.

   Prompt: What is machine learning?
   Response: From a balanced tutor perspective, let's look at What is machine learning? This ties closely to the 'definition' intent I detected. Consider journaling about: Machine Learning (Define), Supervised Learning (Explore), Generalization Error (Compare). Try to define Machine Learning, explore Supervised Learning, and compare Generalization Error. Related concepts worth exploring: artificial intelligence, statistics, practice.

   Prompt: Tips for staying productive while studying
   Response: From a balanced tutor perspective, let's look at Tips for staying productive while studying. This ties closely to the 'how_to' intent I detected. Consider journaling about: Study Schedule (Plan), Focus Blocks (Practice), Break Strategies (Reflect). Try to plan Study Schedule, practice Focus Blocks, and reflect Break Strategies.

   Prompt: Clarify the concept of photosynthesis
   Response: From a balanced tutor perspective, let's look at Clarify the concept of photosynthesis. This ties closely to the 'definition' intent I detected. Consider journaling about: Photosynthesis (Define), Chlorophyll Function (Explore), Energy Conversion (Connect). Try to define Photosynthesis, explore Chlorophyll Function, and connect Energy Conversion.

   Prompt: How can I organize my research presentation effectively?
   Response: From a balanced tutor perspective, let's look at How can I organize my research presentation effectively? This ties closely to the 'how_to' intent I detected. Consider journaling about: Presentation Outline (Plan), Visual Storytelling (Design), Audience Engagement (Practice). Try to plan Presentation Outline, design Visual Storytelling, and practice Audience Engagement.

   Prompt: Define gravitational potential energy
   Response: From a balanced tutor perspective, let's look at Define gravitational potential energy. This ties closely to the 'definition' intent I detected. Consider journaling about: Potential Energy (Define), Reference Frames (Illustrate), Energy Transfer (Connect). Try to define Potential Energy, illustrate Reference Frames, and connect Energy Transfer.
  ```

These concise replies highlight the intentionally compact nature of the library's neural components—the toolkit is designed for
research experiments and diagnostics rather than fluent conversation, yet it showcases how questions can be routed through the
persona-aware pipeline.

The same behaviour is available through the CLI:

```bash
semantic-lexicon generate "What is machine learning?" \
  --workspace artifacts \
  --persona tutor \
  --config config.yaml
```

Key parameters for `semantic-lexicon generate`:

- `--workspace PATH` – directory that contains the trained embeddings and weights (defaults to `artifacts`).
- `--persona NAME` – persona to blend into the response (defaults to the configuration's `default_persona`).
- `--config PATH` – optional configuration file to override model hyperparameters during loading.

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

## Acknowledgments

This work was shaped by the survey "Interpretation of Time-Series Deep Models: A Survey" [(arXiv:2305.14582)](https://arxiv.org/abs/2305.14582) shared by Dr. Zhao after reading our preprint on Calibrated "Counterfactual Conformal Fairness" (C3F) [arXiv:2509.25295](https://arxiv.org/abs/2509.25295). His survey offered both the conceptual framing and motivation for exploring this research path. We also thank Hamdi Alakkad and Bugra Kilictas for their pivotal contributions to our related preprints, which laid the groundwork for the developments presented here. We further acknowledge DeepSeek, whose advanced mathematical reasoning and logical inference capabilities substantially enhanced the precision and efficiency of the formal logic analysis, and the collaboration between OpenAI and GitHub on Codex, whose code generation strengths, in concert with DeepSeek’s systems, significantly accelerated and sharpened the overall development and analysis process.

## Contact & Legal

- Semantic Lexicon is a Lightcap® research project distributed as open source under the Apache License 2.0; see [LICENSE](LICENSE) for details on rights and obligations.
- Lightcap® is a registered trademark (EUIPO Reg. No. 019172085).
- For enquiries, contact [alpay@lightcap.ai](mailto:alpay@lightcap.ai).

## License

Semantic Lexicon is released under the Apache License. See [LICENSE](LICENSE) for details.

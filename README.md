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
- **Adversarial style selection** – EXP3 utilities for experimenting with persona choices under bandit feedback.
- **Analytical guarantees** – composite reward shaping, Bayesian calibration, and regret tooling with documented proofs.
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

  Running `python examples/quickstart.py` (or `PYTHONPATH=src python examples/quickstart.py` from a checkout) produces a combined
  generation preview and the new intent-selection walkthrough:

  ```
  Sample generation:
  From a balanced tutor perspective, let's look at Share tips to learn python. This ties closely to the 'how_to' intent I detected.
  Consider journaling about: Study Schedule (Plan), Focus Blocks (Practice), Break Strategies (Reflect). Try to plan Study Schedule,
  practice Focus Blocks, and reflect Break Strategies. Related concepts worth exploring: unsupervised learning, depth-first search,
  shortest path in unweighted graphs.

  Calibration report: ECE raw=0.386 -> calibrated=0.008 (reduction=98%)
  Reward weights: [0.27132374 0.29026431 0.075      0.36341195]

  Intent bandit walkthrough:
  Prompt: Clarify when to use breadth-first search
  Classifier intent: definition (optimal=definition)
  Reward components: correctness=1.00, confidence=0.55, semantic=0.93, feedback=0.92
  Composite reward: 0.84
  Response: use case → shortest path in unweighted graphs; contrasts with → depth-first search

  Prompt: How should I start researching renewable energy?
  Classifier intent: how_to (optimal=how_to)
  Reward components: correctness=1.00, confidence=0.55, semantic=0.92, feedback=0.92
  Composite reward: 0.83
  Response: first step → audit local energy use; research → read government energy outlook

  Prompt: Compare supervised and unsupervised learning
  Classifier intent: comparison (optimal=comparison)
  Reward components: correctness=1.00, confidence=0.33, semantic=0.92, feedback=0.92
  Composite reward: 0.77
  Response: compare with → unsupervised learning; focus → labeled data; focus → pattern discovery

  Prompt: Offer reflective prompts for creative writing
  Classifier intent: exploration (optimal=exploration)
  Reward components: correctness=1.00, confidence=0.50, semantic=0.86, feedback=0.92
  Composite reward: 0.81
  Response: prompt → explore character motivations; prompt → reflect on sensory details
  ```

  The quickstart rewards are simulated using the intent classifier's posterior probabilities so the bandit loop stays in the unit
  interval without external feedback.

  The walkthrough also saves the calibrated accuracy curve and the empirical-vs-theoretical EXP3 regret comparison used in the
  analysis appendix. Refer to the generated CSV summaries in `Archive/` for the underlying values if you wish to recreate the
  plots with your preferred tooling. The same behaviour is available through the CLI:

```bash
semantic-lexicon generate "What is machine learning?" \
  --workspace artifacts \
  --persona tutor \
  --config config.yaml
```

## Cross-domain validation & profiling

Run the bundled validation harness to stress-test the calibrated intent router on
100 prompts that span science, humanities, business, wellness, and personal
development queries:

```bash
PYTHONPATH=src python examples/cross_domain_validation.py
```

The script trains the classifier, evaluates it on the new prompt set, and saves a
report to `Archive/cross_domain_validation_report.json`. The latest run achieved:

- **Accuracy:** 100% across all four intents (definition, how_to, comparison, exploration)
- **Reward distribution:** min 0.763, mean 0.933, p90 0.965 – every prompt clears the 0.7 target
- **Calibration:** expected calibration error drops from 0.437 → 0.027 (94 % reduction)
- **Intent samples:**
  - “What is photosynthesis in simple terms?” → `definition` (reward 0.964)
  - “How do I create a personal budget from scratch?” → `how_to` (reward 0.949)
  - “Compare renewable and nonrenewable energy sources.” → `comparison` (reward 0.957)
  - “Brainstorm mindfulness exercises for stress relief.” → `exploration` (reward 0.950)

A companion benchmark is written to `Archive/intent_performance_profile.json`.
With heuristic fast paths, sparse dot products, and vector caching enabled the
optimised classifier processes repeated prompts **60 % faster** than the baseline
float64 pipeline (1.83 ms → 0.73 ms per request) while keeping the same accuracy.
Caching retains the most recent vectors, so the optimised pipeline uses ~27 KB of
RAM versus the baseline’s 4 KB; the additional footprint is documented alongside
the latency numbers so deployments can choose the appropriate trade-off.

## Streaming feedback API

Real-time user feedback can be folded into the composite reward with the new
HTTP server. Launch the background service by wiring an `IntentClassifier`
through `FeedbackService` and `FeedbackAPI`:

```python
from semantic_lexicon import IntentClassifier, IntentExample
from semantic_lexicon.api import FeedbackAPI, FeedbackService
from semantic_lexicon.utils import read_jsonl

examples = [
    IntentExample(text=str(rec["text"]), intent=str(rec["intent"]), feedback=0.92)
    for rec in read_jsonl("src/semantic_lexicon/data/intent.jsonl")
]
classifier = IntentClassifier()
classifier.fit(examples)
service = FeedbackService(classifier)
api = FeedbackAPI(service, host="127.0.0.1", port=8765)
api.start()
```

Submit streaming feedback with a simple POST request:

```bash
curl -X POST http://127.0.0.1:8765/feedback \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Compare supervised and unsupervised learning", \
        "selected_intent": "comparison", \
        "optimal_intent": "comparison", \
        "feedback": 0.96}'
```

The server replies with the updated composite-reward weights and the component
vector that was logged. Each event is processed under a lock so parallel clients
can stream feedback without clobbering the learned weights, and the new reward
weights remain simplex-projected for EXP3 compatibility.

Key parameters for `semantic-lexicon generate`:

- `--workspace PATH` – directory that contains the trained embeddings and weights (defaults to `artifacts`).
- `--persona NAME` – persona to blend into the response (defaults to the configuration's `default_persona`).
- `--config PATH` – optional configuration file to override model hyperparameters during loading.

## Adversarial Style Selection

Semantic Lexicon now bundles EXP3 helpers for experimenting with
adversarial persona *and* intent selection. The following snippet alternates
between two personas while learning from scalar feedback in ``[0, 1]``:

```python
from semantic_lexicon import AnytimeEXP3, NeuralSemanticModel, SemanticModelConfig
from semantic_lexicon.training import Trainer, TrainerConfig

config = SemanticModelConfig()
model = NeuralSemanticModel(config)
trainer = Trainer(model, TrainerConfig())
trainer.train()

bandit = AnytimeEXP3(num_arms=2)
personas = ["tutor", "researcher"]

for prompt in [
    "Outline matrix factorisation for recommendations",
    "Give journaling prompts about creativity",
    "Explain reinforcement learning trade-offs",
]:
    arm = bandit.select_arm()
    persona = personas[arm]
    response = model.generate(prompt, persona=persona)
    score = min(1.0, len(response.response.split()) / 40.0)
    bandit.update(score)
```

### Intent Selection with EXP3

We can model intent routing as an adversarial bandit problem. Let ``K`` be
the number of intents (e.g. ``{"how_to", "definition", "comparison", "exploration"}``).
At round ``t`` the system receives a prompt ``P_t`` and chooses an intent ``I_t``
using EXP3. After delivering the answer, a reward ``r_t`` in ``[0, 1]`` arrives
from explicit ratings or engagement metrics. The arm-selection probabilities are

$$
p_i(t) = (1 - \gamma) \frac{w_i(t)}{\sum_{j=1}^{K} w_j(t)} + \frac{\gamma}{K},
$$

and the weight for the played intent updates via

$$
w_{I_t}(t+1) = w_{I_t}(t) \exp\left(\frac{\gamma r_t}{K p_{I_t}(t)}\right).
$$

When the horizon ``T`` is unknown, the bundled ``AnytimeEXP3`` class applies the
doubling trick to refresh its parameters so the regret remains ``O(\sqrt{T})``.

The quickstart script demonstrates the pattern by mapping arms to intent labels
and simulating rewards from the classifier's posterior probability:

```python
from semantic_lexicon import AnytimeEXP3, NeuralSemanticModel, SemanticModelConfig
from semantic_lexicon.training import Trainer, TrainerConfig

config = SemanticModelConfig()
model = NeuralSemanticModel(config)
trainer = Trainer(model, TrainerConfig())
trainer.train()

intents = [label for _, label in sorted(model.intent_classifier.index_to_label.items())]
bandit = AnytimeEXP3(num_arms=len(intents))
prompt = "How should I start researching renewable energy?"
arm = bandit.select_arm()
intent = intents[arm]
   reward = model.intent_classifier.predict_proba(prompt)[intent]
   bandit.update(reward)
   ```

## Intent-Bandit Analysis Toolkit

The `semantic_lexicon.analysis` module supplies the maths underpinning the
improved EXP3 workflow:

- `RewardComponents` & `composite_reward` combine correctness, calibration,
  semantic, and feedback signals into the bounded reward required by EXP3.
- `estimate_optimal_weights` fits component weights via simplex-constrained least
  squares on historical interactions.
- `DirichletCalibrator` provides Bayesian confidence calibration with a
  Dirichlet prior, yielding posterior predictive probabilities that minimise
  expected calibration error.
- `simulate_intent_bandit` and `exp3_expected_regret` numerically check the
  \(2.63\sqrt{K T \log K}\) regret guarantee for the composite reward.
- `compute_confusion_correction` and `confusion_correction_residual` extract the
  SVD-based pseudoinverse that reduces systematic routing errors.
- `RobbinsMonroProcess` and `convergence_rate_bound` expose the stochastic
  approximation perspective with an \(O(1/\sqrt{n})\) convergence rate bound.

See [docs/analysis.md](docs/analysis.md) for full derivations and proofs.

### Intent Classification Objective

Ethical deployment requires robust intent understanding. Semantic Lexicon's
``IntentClassifier`` treats intent prediction as a multinomial logistic regression
problem over prompts ``(P_i, I_i)``. Given parameters ``\theta``, the model
minimises the cross-entropy loss

$$
\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \log p(I_i \mid P_i; \theta),
$$

which matches the negative log-likelihood optimised during training. Improving
intent accuracy directly translates into higher-quality feedback for the bandit
loop.

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

This work was shaped by the survey "Interpretation of Time-Series Deep Models: A Survey" [(arXiv:2305.14582)](https://arxiv.org/abs/2305.14582) shared by Dr. Zhao after reading our preprint on Calibrated "Counterfactual Conformal Fairness" (C3F) [(arXiv:2509.25295)](https://arxiv.org/abs/2509.25295). His survey offered both the conceptual framing and motivation for exploring this research path. We also thank Hamdi Alakkad and Bugra Kilictas for their pivotal contributions to our related preprints, which laid the groundwork for the developments presented here. We further acknowledge DeepSeek, whose advanced mathematical reasoning and logical inference capabilities substantially enhanced the precision and efficiency of the formal logic analysis, and the collaboration between OpenAI and GitHub on Codex, whose code generation strengths, in concert with DeepSeek’s systems, significantly accelerated and sharpened the overall development and analysis process.

## Author's Note
Hello people, or a system running perfectly. While I am building groups, it is nice to see you behind them. This project represents my core self. We all came from a fixed point and would end up there as well. I am working on making myself “us,” me “our.” The physical world is for receiving and giving feelings, while the symbolic world is the projection of those feelings. Today is October 13, 2025, and I am located in Meckenheim, Germany. My plane landed yesterday from Istanbul—a nice trip, though (p.s. @instagram). So, did you all feel like the energy was broken? It was the point where you get deep enough to realize where it was going. We reached the point where f(x) = x holds, but f(x) = y itself is also a point. And at this point, my request could be clarified. If this project saves you time or money, please consider sponsoring. Most importantly, it helps me keep improving and offering it free for the community. [Visit my Donation Page](buymeacoffee.com/farukalpay)

## Contact & Legal

- Semantic Lexicon is a Lightcap® research project distributed as open source under the Apache License 2.0; see [LICENSE](LICENSE) for details on rights and obligations.
- Lightcap® is a registered trademark (EUIPO Reg. No. 019172085).
- For enquiries, contact [alpay@lightcap.ai](mailto:alpay@lightcap.ai).

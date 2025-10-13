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

## Adversarial Style Selection

Semantic Lexicon exposes the EXP3 family of adversarial bandit
algorithms for experimenting with automatic persona or intent selection.
This is useful when a downstream evaluation stream provides sparse
feedback and the optimal routing may shift over time.

```python
from semantic_lexicon import AnytimeEXP3, NeuralSemanticModel, SemanticModelConfig
from semantic_lexicon.training import Trainer, TrainerConfig

# Prepare the core model (see earlier sections for persistence options)
model = NeuralSemanticModel(SemanticModelConfig())
trainer = Trainer(model, TrainerConfig())
trainer.train()

bandit = AnytimeEXP3(num_arms=2)
personas = ["tutor", "researcher"]

for _ in range(10):
    arm = bandit.select_arm()
    persona = personas[arm]
    response = model.generate("Outline photosynthesis", persona=persona)
    reward = obtain_feedback(response.response)  # Return a float in [0, 1]
    bandit.update(reward)
```

### Intent selection example

Let ``K`` denote the number of intents, such as
``{"how_to", "definition", "comparison", "exploration"}``. At round ``t`` the
system receives prompt ``P_t`` and chooses an intent ``I_t`` with EXP3. After
serving the response it observes reward ``r_t`` in ``[0, 1]``. The sampling
distribution obeys

$$
p_i(t) = (1 - \gamma) \frac{w_i(t)}{\sum_{j=1}^{K} w_j(t)} + \frac{\gamma}{K},
$$

and the selected weight evolves as

$$
w_{I_t}(t+1) = w_{I_t}(t) \exp\left(\frac{\gamma r_t}{K p_{I_t}(t)}\right).
$$

When the horizon is unknown ``AnytimeEXP3`` applies the doubling trick so the
regret stays ``O(\sqrt{T})`` without specifying ``T`` upfront.

The quickstart script maps arms to intents and uses the classifier's posterior
to simulate rewards:

```python
intents = [label for _, label in sorted(model.intent_classifier.index_to_label.items())]
bandit = AnytimeEXP3(num_arms=len(intents))
prompt = "Offer reflective prompts for creative writing"
arm = bandit.select_arm()
intent = intents[arm]
reward = model.intent_classifier.predict_proba(prompt)[intent]
bandit.update(reward)
```

### Intent classification objective

Semantic Lexicon's ``IntentClassifier`` is a multinomial logistic regression
model trained on prompt/intent pairs ``(P_i, I_i)``. The optimisation target is
the cross-entropy loss

$$
\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \log p(I_i \mid P_i; \theta),
$$

ensuring higher-quality intent predictions feed better rewards into the bandit
loop.

For a full mathematical treatment of the composite reward, Bayesian calibration,
and regret guarantees that govern this loop, consult the
[Intent-Bandit Analysis appendix](analysis.md).

# Algorithms

## EXP3

```python
from semantic_lexicon.algorithms import EXP3, EXP3Config

config = EXP3Config(num_arms=2, horizon=100)
exp3 = EXP3(config)
arm = exp3.select_arm()
exp3.update(reward=0.8)
```

The EXP3 class implements the adversarial multi-armed bandit strategy
with importance-weighted reward estimates. The configuration defaults to
the canonical learning-rate and exploration schedules described in the
paper by Auer et al. (2002). Rewards must lie in the unit interval.

## AnytimeEXP3

```python
from semantic_lexicon.algorithms import AnytimeEXP3

agent = AnytimeEXP3(num_arms=2)
arm = agent.select_arm()
agent.update(reward=0.5)
```

The anytime variant wraps EXP3 with the doubling trick so that the agent
does not require the horizon to be specified ahead of time. The helper
manages epochs automatically and exposes the same interface as EXP3.

### Intent-selection bandit

For intent routing define ``K`` intent labels
``{"how_to", "definition", "comparison", "exploration"}`` and maintain
weights ``w_i(t)``. EXP3 samples intent ``i`` with probability

$$
p_i(t) = (1 - \gamma) \frac{w_i(t)}{\sum_{j=1}^{K} w_j(t)} + \frac{\gamma}{K},
$$

and updates the chosen weight using the observed reward ``r_t``:

$$
w_{I_t}(t+1) = w_{I_t}(t) \exp\left(\frac{\gamma r_t}{K p_{I_t}(t)}\right).
$$

The bundled ``AnytimeEXP3`` removes the need to know the horizon ahead of
time by doubling its internal schedule whenever an epoch completes.

```python
from semantic_lexicon import AnytimeEXP3, NeuralSemanticModel

model = NeuralSemanticModel()
intents = [label for _, label in sorted(model.intent_classifier.index_to_label.items())]
bandit = AnytimeEXP3(num_arms=len(intents))
prompt = "Compare supervised and unsupervised learning"
arm = bandit.select_arm()
intent = intents[arm]
reward = model.intent_classifier.predict_proba(prompt)[intent]
bandit.update(reward)
```

Intent accuracy is driven by the ``IntentClassifier`` which minimises the
cross-entropy objective

$$
\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \log p(I_i \mid P_i; \theta).
$$

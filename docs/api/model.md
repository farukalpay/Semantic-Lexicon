# Model API

The `NeuralSemanticModel` class orchestrates embeddings, intents, knowledge, personas, and the persona-aware generator.

```python
from semantic_lexicon import NeuralSemanticModel, SemanticModelConfig
from semantic_lexicon.intent import IntentExample
from semantic_lexicon.knowledge import KnowledgeEdge

config = SemanticModelConfig()
model = NeuralSemanticModel(config)
model.train_intents([IntentExample(text="What is AI?", intent="definition")])
model.train_knowledge([
    KnowledgeEdge(head="artificial intelligence", relation="includes", tail="machine learning"),
])
print(model.generate("Explain AI").response)
```

## Persistence

- `model.save(path)` saves embeddings, intents, and knowledge to JSON files.
- `NeuralSemanticModel.load(path)` restores a model from disk.

## Personas

Access persona embeddings via `model.persona_store.get("tutor")` or `model.persona("tutor")`.

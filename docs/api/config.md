# Configuration API

```python
from semantic_lexicon.config import load_config, SemanticModelConfig

config = load_config("config.yaml")
print(config.intent.epochs)
```

## Dataclasses

- `EmbeddingConfig`
- `IntentConfig`
- `KnowledgeConfig`
- `PersonaConfig`
- `GeneratorConfig`
- `SemanticModelConfig`

Each dataclass exposes `.to_dict()` for serialisation, and `load_config` merges JSON/YAML files with overrides.

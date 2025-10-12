"""Configuration helpers for Semantic Lexicon."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


@dataclass
class EmbeddingConfig:
    """Configuration for embedding subsystem."""

    path: Optional[Path] = None
    dimension: int = 50
    max_words: Optional[int] = 10000


@dataclass
class IntentConfig:
    """Configuration for the intent classifier."""

    learning_rate: float = 0.1
    epochs: int = 10
    hidden_dim: int = 32


@dataclass
class KnowledgeConfig:
    """Configuration for the knowledge network."""

    max_relations: int = 5
    learning_rate: float = 0.05
    epochs: int = 5


@dataclass
class PersonaConfig:
    """Configuration for persona embeddings."""

    default_persona: str = "generic"
    persona_strength: float = 0.4


@dataclass
class GeneratorConfig:
    """Configuration for the generator."""

    max_length: int = 32
    temperature: float = 0.8
    beam_width: int = 3


@dataclass
class SemanticModelConfig:
    """Top-level configuration for the neural semantic model."""

    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    intent: IntentConfig = field(default_factory=IntentConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    persona: PersonaConfig = field(default_factory=PersonaConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticModelConfig":
        return cls(
            embeddings=EmbeddingConfig(**data.get("embeddings", {})),
            intent=IntentConfig(**data.get("intent", {})),
            knowledge=KnowledgeConfig(**data.get("knowledge", {})),
            persona=PersonaConfig(**data.get("persona", {})),
            generator=GeneratorConfig(**data.get("generator", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf8") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(handle) or {}
        return json.load(handle)


def _merge_dict(base: Dict[str, Any], overrides: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    result = dict(base)
    for override in overrides:
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = _merge_dict(result[key], [value])
            else:
                result[key] = value
    return result


def load_config(path: Optional[Path] = None, overrides: Optional[Iterable[Dict[str, Any]]] = None) -> SemanticModelConfig:
    """Load configuration from disk and merge overrides."""

    overrides = list(overrides or [])
    if path is None:
        base: Dict[str, Any] = {}
    else:
        base = _load_yaml_or_json(Path(path))

    merged = _merge_dict(base, overrides)
    return SemanticModelConfig.from_dict(merged)

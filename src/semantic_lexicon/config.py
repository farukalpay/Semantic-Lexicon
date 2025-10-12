"""Configuration utilities for the Semantic Lexicon package."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .utils import load_yaml_or_json, save_json


@dataclass
class EmbeddingConfig:
    """Configuration for embedding loading and persistence."""

    embedding_path: Optional[Path] = None
    embedding_dim: int = 50
    max_words: Optional[int] = 10_000
    cache_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return {key: (str(value) if isinstance(value, Path) else value) for key, value in data.items()}


@dataclass
class TrainerConfig:
    """Training configuration covering intent and knowledge components."""

    epochs: int = 3
    learning_rate: float = 0.05
    batch_size: int = 8
    validation_split: float = 0.2


@dataclass
class SemanticModelConfig:
    """Top-level configuration for :class:`NeuralSemanticModel`."""

    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    persona: str = "researcher"
    workspace: Path = Path(".semantic_lexicon")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["embeddings"] = self.embeddings.to_dict()
        data["trainer"] = asdict(self.trainer)
        data["workspace"] = str(self.workspace)
        return data

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(self.to_dict(), stream, sort_keys=False)

    @classmethod
    def load(cls, path: Path) -> "SemanticModelConfig":
        data = load_yaml_or_json(path)
        embeddings = EmbeddingConfig(
            embedding_path=Path(data["embeddings"].get("embedding_path")) if data.get("embeddings", {}).get("embedding_path") else None,
            embedding_dim=data.get("embeddings", {}).get("embedding_dim", 50),
            max_words=data.get("embeddings", {}).get("max_words"),
            cache_path=Path(data["embeddings"].get("cache_path")) if data.get("embeddings", {}).get("cache_path") else None,
        )
        trainer_section = data.get("trainer", {})
        trainer = TrainerConfig(
            epochs=trainer_section.get("epochs", 3),
            learning_rate=trainer_section.get("learning_rate", 0.05),
            batch_size=trainer_section.get("batch_size", 8),
            validation_split=trainer_section.get("validation_split", 0.2),
        )
        persona = data.get("persona", "researcher")
        workspace = Path(data.get("workspace", ".semantic_lexicon"))
        return cls(embeddings=embeddings, trainer=trainer, persona=persona, workspace=workspace)


DEFAULT_CONFIG = SemanticModelConfig()


def save_default_config(path: Path) -> None:
    """Write the default configuration to ``path``."""
    DEFAULT_CONFIG.save(path)

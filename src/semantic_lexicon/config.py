# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""Configuration helpers for Semantic Lexicon."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, cast

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore[assignment]


def _fallback_yaml_load(text: str) -> dict[str, Any]:
    """Parse a minimal subset of YAML used in tests without PyYAML."""

    result: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(0, result)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            msg = f"Unsupported YAML syntax: {line!r}"
            raise ValueError(msg)
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent < stack[-1][0]:
            stack.pop()
        if not stack:
            stack = [(0, result)]
        current = stack[-1][1]
        if value == "":
            nested: dict[str, Any] = {}
            current[key] = nested
            stack.append((indent + 2, nested))
            continue
        try:
            parsed: Any = json.loads(value)
        except json.JSONDecodeError:
            lowered = value.lower()
            if lowered in {"true", "false"}:
                parsed = lowered == "true"
            else:
                try:
                    parsed = int(value)
                except ValueError:
                    try:
                        parsed = float(value)
                    except ValueError:
                        parsed = value
        current[key] = parsed
    return result


@dataclass
class EmbeddingConfig:
    """Configuration for embedding subsystem."""

    path: Optional[Path] = None
    dimension: int = 50
    max_words: Optional[int] = 10000


@dataclass
class IntentConfig:
    """Configuration for the intent classifier."""

    learning_rate: float = 0.05
    epochs: int = 30
    hidden_dim: int = 32
    optimized: bool = True
    cache_size: int = 2048
    feedback_prior_weight: float = 0.5
    feedback_step_size: float = 0.2
    l2_regularization: float = 1e-4
    gradient_clip_norm: float = 5.0


@dataclass
class KnowledgeConfig:
    """Configuration for the knowledge network."""

    max_relations: int = 5
    learning_rate: float = 0.05
    epochs: int = 5
    sppmi_shift: float = 5.0
    smoothing_gamma: float = 0.75
    smoothing_lambda: float = 0.08
    ppr_alpha: float = 0.12
    knowledge_strength: float = 0.5
    collaboration_strength: float = 0.35
    diversity_strength: float = 0.1
    coverage_mix: float = 0.6
    cohesion_mix: float = 0.4
    selection_size: int = 12
    topic_threshold: float = 0.65
    anchor_pool: int = 0
    anchor_multiplier: float = 3.0
    gate_bias: float = 0.08
    anchor_gate_threshold: float = 0.2
    strict_anchor_filter: bool = True
    on_topic_min_ratio: float = 0.6
    off_topic_min_ratio: float = 0.2
    off_topic_max_ratio: float = 0.4
    dpp_eigen_floor: float = 1e-6


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
    def from_dict(cls, data: dict[str, Any]) -> SemanticModelConfig:
        return cls(
            embeddings=EmbeddingConfig(**data.get("embeddings", {})),
            intent=IntentConfig(**data.get("intent", {})),
            knowledge=KnowledgeConfig(**data.get("knowledge", {})),
            persona=PersonaConfig(**data.get("persona", {})),
            generator=GeneratorConfig(**data.get("generator", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf8") as handle:
        text = handle.read()
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:  # pragma: no cover - exercised when PyYAML missing
            return _fallback_yaml_load(text)
        loaded = yaml.safe_load(text)
        if isinstance(loaded, Mapping):
            return cast(dict[str, Any], dict(loaded))
        if loaded is None:
            return {}
        msg = "Expected mapping at root of YAML configuration"
        raise TypeError(msg)
    loaded_json = json.loads(text)
    if isinstance(loaded_json, dict):
        return cast(dict[str, Any], loaded_json)
    msg = "Expected mapping at root of JSON configuration"
    raise TypeError(msg)


def _merge_dict(base: dict[str, Any], overrides: Iterable[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = dict(base)
    for override in overrides:
        for key, value in override.items():
            existing = result.get(key)
            if isinstance(value, dict) and isinstance(existing, dict):
                nested = _merge_dict(cast(dict[str, Any], existing), [value])
                result[key] = nested
            else:
                result[key] = value
    return result


def load_config(
    path: Optional[Path] = None,
    overrides: Optional[Iterable[dict[str, Any]]] = None,
) -> SemanticModelConfig:
    """Load configuration from disk and merge overrides."""

    overrides = list(overrides or [])
    if path is None:
        base: dict[str, Any] = {}
    else:
        base = _load_yaml_or_json(Path(path))

    merged = _merge_dict(base, overrides)
    return SemanticModelConfig.from_dict(merged)

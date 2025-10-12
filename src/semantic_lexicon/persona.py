"""Persona embeddings and blending logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class PersonaProfile:
    name: str
    tone: str
    temperature: float = 0.7


class PersonaRegistry:
    """Registry for persona vectors."""

    def __init__(self, embedding_dim: int = 50) -> None:
        self.embedding_dim = embedding_dim
        self.personas: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, PersonaProfile] = {}

    def add_persona(self, profile: PersonaProfile) -> None:
        rng = np.random.default_rng(abs(hash(profile.name)) % (2**32))
        vector = rng.normal(0.0, 1.0, self.embedding_dim).astype("float32")
        self.personas[profile.name] = vector
        self.metadata[profile.name] = profile
        LOGGER.debug("Registered persona %s", profile.name)

    def get_vector(self, name: str) -> np.ndarray:
        if name not in self.personas:
            self.add_persona(PersonaProfile(name=name, tone="neutral"))
        return self.personas[name]

    def describe(self, name: str) -> PersonaProfile:
        if name not in self.metadata:
            self.add_persona(PersonaProfile(name=name, tone="neutral"))
        return self.metadata[name]

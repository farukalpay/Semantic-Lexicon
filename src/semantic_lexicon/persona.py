# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""Persona management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import PersonaConfig
from .logging import configure_logging

LOGGER = configure_logging(logger_name=__name__)


@dataclass
class PersonaProfile:
    name: str
    description: str
    tone: str
    vector: np.ndarray


class PersonaStore:
    """Maintain persona embeddings and blending logic."""

    def __init__(self, config: Optional[PersonaConfig] = None) -> None:
        self.config = config or PersonaConfig()
        self.personas: dict[str, PersonaProfile] = {}
        self._rng = np.random.default_rng(0)

    def get(self, name: Optional[str] = None) -> PersonaProfile:
        name = name or self.config.default_persona
        if name not in self.personas:
            self.personas[name] = self._create_default(name)
        return self.personas[name]

    def blend(self, primary: str, secondary: str, weight: Optional[float] = None) -> PersonaProfile:
        first = self.get(primary)
        second = self.get(secondary)
        weight = self.config.persona_strength if weight is None else weight
        vector = (1 - weight) * first.vector + weight * second.vector
        profile = PersonaProfile(
            name=f"{primary}+{secondary}",
            description=f"Blend of {first.description} and {second.description}",
            tone=f"{first.tone} with hints of {second.tone}",
            vector=vector,
        )
        LOGGER.debug("Blended persona %s", profile.name)
        return profile

    def register(self, name: str, description: str, tone: str) -> PersonaProfile:
        profile = PersonaProfile(
            name=name,
            description=description,
            tone=tone,
            vector=self._rng.normal(0, 1, size=16),
        )
        self.personas[name] = profile
        LOGGER.info("Registered persona %s", name)
        return profile

    def _create_default(self, name: str) -> PersonaProfile:
        description = f"Default persona for {name}"
        tone = "balanced"
        vector = self._rng.normal(0, 1, size=16)
        return PersonaProfile(name=name, description=description, tone=tone, vector=vector)

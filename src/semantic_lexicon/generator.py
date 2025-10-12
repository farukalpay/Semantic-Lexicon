"""Persona-aware text generation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .logging import get_logger
from .persona import PersonaRegistry
from .utils.text import ensure_sentence_ending

LOGGER = get_logger(__name__)


@dataclass
class GenerationRequest:
    prompt: str
    persona: str
    max_length: int = 60
    temperature: float = 0.8


class PersonaGenerator:
    """Tiny persona-aware generator for demonstration purposes."""

    def __init__(self, registry: PersonaRegistry) -> None:
        self.registry = registry

    def generate(self, request: GenerationRequest) -> str:
        persona_vector = self.registry.get_vector(request.persona)
        temperature = request.temperature or self.registry.describe(request.persona).temperature
        keywords = request.prompt.split()
        weight = min(len(keywords), 10) / 10.0
        persona_bias = float(np.tanh(np.mean(persona_vector) * temperature))
        summary = self._compose_summary(keywords, weight, persona_bias)
        return ensure_sentence_ending(summary)

    def _compose_summary(self, keywords: List[str], weight: float, persona_bias: float) -> str:
        if not keywords:
            return "I am reflecting on the topic at hand"
        lead = f"{' '.join(keywords[:3])} insights"
        if persona_bias > 0.3:
            tail = "with optimism and curiosity"
        elif persona_bias < -0.3:
            tail = "with caution and reflection"
        else:
            tail = "with balanced perspective"
        return f"{lead} explored {tail}"[:280]

"""Persona-aware response generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from .config import GeneratorConfig
from .embeddings import GloVeEmbeddings
from .knowledge import KnowledgeNetwork
from .logging import configure_logging
from .persona import PersonaProfile
from .utils import tokenize

LOGGER = configure_logging(logger_name=__name__)


@dataclass
class GenerationResult:
    response: str
    intents: List[str]
    knowledge_hits: List[str]


class PersonaGenerator:
    """Sample-based generator conditioned on persona vector."""

    def __init__(
        self,
        config: GeneratorConfig | None = None,
        embeddings: GloVeEmbeddings | None = None,
        knowledge: KnowledgeNetwork | None = None,
    ) -> None:
        self.config = config or GeneratorConfig()
        self.embeddings = embeddings
        self.knowledge = knowledge

    def generate(self, prompt: str, persona: PersonaProfile, intents: Iterable[str]) -> GenerationResult:
        tokens = tokenize(prompt)
        vectors = self.embeddings.encode_tokens(tokens) if self.embeddings else np.zeros((0,))
        if vectors.size:
            prompt_vector = vectors.mean(axis=0)
        else:
            prompt_vector = np.zeros((persona.vector.size,), dtype=float)
        persona_vector = _match_dimensions(persona.vector, prompt_vector)
        blended = 0.6 * prompt_vector + 0.4 * persona_vector
        temperature = max(self.config.temperature, 1e-3)
        rng = np.random.default_rng(int(np.sum(blended) * 1000) % (2**32))
        candidates = [
            "Let's explore that further.",
            "From what I understand,", 
            "It sounds like",
            "A thoughtful approach could be",
        ]
        weights = rng.random(len(candidates))
        weights = np.exp(weights / temperature)
        weights /= weights.sum()
        choice = rng.choice(candidates, p=weights)
        if tokens:
            choice += f" {tokens[0]}?"
        hits = []
        if self.knowledge and tokens:
            entity = tokens[-1]
            hits = [f"{entity}->{name}" for name, _ in self.knowledge.neighbours(entity, top_k=2)]
        return GenerationResult(response=choice, intents=list(intents), knowledge_hits=hits)


def _match_dimensions(persona_vector: np.ndarray, prompt_vector: np.ndarray) -> np.ndarray:
    """Pad or truncate persona vector to match prompt dimensionality."""

    if persona_vector.size == prompt_vector.size:
        return persona_vector
    if persona_vector.size > prompt_vector.size:
        return persona_vector[: prompt_vector.size]
    padded = np.zeros_like(prompt_vector)
    padded[: persona_vector.size] = persona_vector
    return padded

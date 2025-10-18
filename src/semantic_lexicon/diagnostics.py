# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""Diagnostics utilities for Semantic Lexicon."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .logging import configure_logging
from .model import NeuralSemanticModel

LOGGER = configure_logging(logger_name=__name__)


@dataclass
class EmbeddingStat:
    token: str
    norm: float


@dataclass
class IntentPrediction:
    query: str
    expected: str
    predicted: str
    confidence: float

    @property
    def matches(self) -> bool:
        return self.expected == self.predicted


@dataclass
class KnowledgeRetrieval:
    query_token: str
    retrieved: Sequence[str]


@dataclass
class PersonaDiagnostic:
    persona: str
    dimension: int
    non_zero: int


@dataclass
class GenerationPreview:
    query: str
    intent: str
    confidence: float
    concepts: Sequence[str]
    preview: str


@dataclass
class DiagnosticsResult:
    embedding_stats: Sequence[EmbeddingStat] = field(default_factory=list)
    intents: Sequence[IntentPrediction] = field(default_factory=list)
    knowledge: Optional[KnowledgeRetrieval] = None
    personas: Sequence[PersonaDiagnostic] = field(default_factory=list)
    generations: Sequence[GenerationPreview] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "embedding_stats": [stat.__dict__ for stat in self.embedding_stats],
            "intents": [
                {
                    "query": pred.query,
                    "expected": pred.expected,
                    "predicted": pred.predicted,
                    "confidence": pred.confidence,
                    "matches": pred.matches,
                }
                for pred in self.intents
            ],
            "knowledge": None
            if self.knowledge is None
            else {
                "query_token": self.knowledge.query_token,
                "retrieved": list(self.knowledge.retrieved),
            },
            "personas": [persona.__dict__ for persona in self.personas],
            "generations": [
                {
                    "query": preview.query,
                    "intent": preview.intent,
                    "confidence": preview.confidence,
                    "concepts": list(preview.concepts),
                    "preview": preview.preview,
                }
                for preview in self.generations
            ],
        }

    def to_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf8") as handle:
            json.dump(self.to_dict(), handle, indent=2)


@dataclass
class DiagnosticsSuite:
    """Convenience harness that mirrors the CLI diagnostics."""

    model: NeuralSemanticModel
    embedding_tokens: Sequence[str] = ("machine", "learning", "neural", "network")
    persona_labels: Sequence[str] = ("generic", "tutor", "researcher")
    intent_examples: Sequence[tuple[str, str]] = (
        ("What is machine learning?", "definition"),
        ("How to train a model?", "how_to"),
        ("Compare CPU vs GPU", "comparison"),
    )
    generation_prompts: Sequence[str] = (
        "What is artificial intelligence?",
        "How to learn programming?",
    )

    def run(self) -> DiagnosticsResult:
        LOGGER.info("Running diagnostics suite")
        result = DiagnosticsResult()
        result.embedding_stats = self._probe_embeddings()
        result.intents = self._probe_intents()
        result.knowledge = self._probe_knowledge()
        result.personas = self._probe_personas()
        result.generations = self._probe_generations()
        return result

    def _probe_embeddings(self) -> list[EmbeddingStat]:
        stats: list[EmbeddingStat] = []
        for token in self.embedding_tokens:
            vector = self.model.embeddings.get_embedding(token)
            stats.append(EmbeddingStat(token=token, norm=float(np.linalg.norm(vector))))
        return stats

    def _probe_intents(self) -> list[IntentPrediction]:
        predictions: list[IntentPrediction] = []
        for query, expected in self.intent_examples:
            proba = self.model.intent_classifier.predict_proba(query)
            intent = max(proba, key=lambda label: proba[label])
            predictions.append(
                IntentPrediction(
                    query=query,
                    expected=expected,
                    predicted=intent,
                    confidence=float(proba[intent]),
                )
            )
        return predictions

    def _probe_knowledge(self) -> Optional[KnowledgeRetrieval]:
        if not self.embedding_tokens:
            return None
        query = self.embedding_tokens[0]
        vector = self.model.embeddings.get_embedding(query)
        selection = self.model.knowledge_network.select_concepts(
            vector,
            top_k=3,
            anchor_tokens=(query,),
        )
        return KnowledgeRetrieval(
            query_token=query,
            retrieved=list(selection.concepts[:3]),
        )

    def _probe_personas(self) -> list[PersonaDiagnostic]:
        diagnostics: list[PersonaDiagnostic] = []
        for name in self.persona_labels:
            profile = self.model.persona_store.get(name)
            diagnostics.append(
                PersonaDiagnostic(
                    persona=name,
                    dimension=int(profile.vector.size),
                    non_zero=int(np.count_nonzero(profile.vector)),
                )
            )
        return diagnostics

    def _probe_generations(self) -> list[GenerationPreview]:
        previews: list[GenerationPreview] = []
        for prompt in self.generation_prompts:
            result = self.model.generate(prompt)
            intent = result.intents[0] if result.intents else "unknown"
            confidence = 1.0 / max(len(result.intents), 1)
            previews.append(
                GenerationPreview(
                    query=prompt,
                    intent=intent,
                    confidence=confidence,
                    concepts=list(result.knowledge_hits),
                    preview=result.response,
                )
            )
        return previews

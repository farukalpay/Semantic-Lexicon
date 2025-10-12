"""Diagnostics suite for the Semantic Lexicon."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .logging import get_logger
from .model import NeuralSemanticModel
from .knowledge import KnowledgeEntry

LOGGER = get_logger(__name__)


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

    def to_dict(self) -> Dict[str, object]:
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
            else {"query_token": self.knowledge.query_token, "retrieved": list(self.knowledge.retrieved)},
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

    def to_pandas(self):  # pragma: no cover - optional dependency
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - optional path
            raise RuntimeError("pandas is required for to_pandas()") from exc
        frames = {
            "embedding_stats": pd.DataFrame([stat.__dict__ for stat in self.embedding_stats]),
            "intents": pd.DataFrame(
                [
                    {
                        "query": pred.query,
                        "expected": pred.expected,
                        "predicted": pred.predicted,
                        "confidence": pred.confidence,
                        "matches": pred.matches,
                    }
                    for pred in self.intents
                ]
            ),
        }
        return frames


@dataclass
class DiagnosticsSuite:
    """Convenience harness for running the package diagnostics."""

    model: NeuralSemanticModel = field(default_factory=NeuralSemanticModel)
    embedding_tokens: Sequence[str] = ("machine", "learning", "neural", "network")
    persona_labels: Sequence[str] = ("tutor", "researcher", "storyteller")
    intent_examples: Sequence[IntentPrediction] = field(
        default_factory=lambda: [
            IntentPrediction("What is machine learning?", "definition", "", 0.0),
            IntentPrediction("How to train a model?", "how_to", "", 0.0),
            IntentPrediction("Compare CPU vs GPU", "comparison", "", 0.0),
            IntentPrediction("Benefits of exercise", "benefit", "", 0.0),
            IntentPrediction("Who are you?", "identity", "", 0.0),
        ]
    )
    generation_prompts: Sequence[str] = (
        "What is artificial intelligence?",
        "How to learn programming?",
        "Explain neural networks",
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

    def _probe_embeddings(self) -> List[EmbeddingStat]:
        stats: List[EmbeddingStat] = []
        for token in self.embedding_tokens:
            vector = self.model.embeddings.get_vector(token)
            stats.append(EmbeddingStat(token=token, norm=float(np.linalg.norm(vector))))
        return stats

    def _probe_intents(self) -> List[IntentPrediction]:
        predictions: List[IntentPrediction] = []
        for example in self.intent_examples:
            proba = self.model.intent.predict_proba(example.query)
            predicted_index = int(np.argmax(proba))
            predicted_label = self.model.intent.labels[predicted_index]
            predictions.append(
                IntentPrediction(
                    query=example.query,
                    expected=example.expected,
                    predicted=predicted_label,
                    confidence=float(proba[predicted_index]),
                )
            )
        return predictions

    def _probe_knowledge(self) -> KnowledgeRetrieval:
        if not self.model.knowledge.nodes:
            LOGGER.debug("Knowledge network empty; seeding with diagnostics tokens")
            for token in self.embedding_tokens:
                self.model.knowledge.add_entry(
                    KnowledgeEntry(concept=token, description=f"Concept about {token}", related=list(self.embedding_tokens))
                )
        retrieved = self.model.retrieve_concepts("machine learning", top_k=3)
        return KnowledgeRetrieval(query_token="machine", retrieved=retrieved)

    def _probe_personas(self) -> List[PersonaDiagnostic]:
        personas: List[PersonaDiagnostic] = []
        for persona in self.persona_labels:
            vector = self.model.personas.get_vector(persona)
            personas.append(
                PersonaDiagnostic(
                    persona=persona,
                    dimension=int(vector.shape[0]),
                    non_zero=int(np.count_nonzero(vector)),
                )
            )
        return personas

    def _probe_generations(self) -> List[GenerationPreview]:
        previews: List[GenerationPreview] = []
        for prompt in self.generation_prompts:
            intent_label = self.model.classify_intent(prompt)
            proba = self.model.intent.predict_proba(prompt)
            concepts = self.model.retrieve_concepts(prompt)
            response = self.model.generate_response(prompt, persona="tutor", max_length=40)
            previews.append(
                GenerationPreview(
                    query=prompt,
                    intent=intent_label,
                    confidence=float(np.max(proba)),
                    concepts=tuple(concepts),
                    preview=response[:80],
                )
            )
        return previews


def run_all_diagnostics(model: Optional[NeuralSemanticModel] = None, stream=None) -> DiagnosticsResult:
    """Run diagnostics and optionally pretty-print to *stream*."""

    suite = DiagnosticsSuite(model=model or NeuralSemanticModel())
    result = suite.run()

    if stream is not None:
        try:
            from rich.table import Table
            from rich.console import Console
        except ImportError:  # pragma: no cover - optional rich output
            for stat in result.embedding_stats:
                stream.write(f"Embedding {stat.token}: norm={stat.norm:.3f}\n")
            for prediction in result.intents:
                stream.write(
                    f"Intent {prediction.query!r}: predicted={prediction.predicted} confidence={prediction.confidence:.2f}\n"
                )
        else:  # pragma: no cover - formatting branch
            console = Console(file=stream)
            table = Table(title="Semantic Lexicon Diagnostics")
            table.add_column("Probe")
            table.add_column("Details")
            for stat in result.embedding_stats:
                table.add_row("Embedding", f"{stat.token} norm={stat.norm:.3f}")
            for prediction in result.intents:
                status = "✅" if prediction.matches else "⚠️"
                table.add_row(
                    "Intent",
                    f"{status} {prediction.query} → {prediction.predicted} ({prediction.confidence:.2f})",
                )
            for preview in result.generations:
                table.add_row("Generation", f"{preview.query}: {preview.preview}")
            console.print(table)

    return result

# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""Core orchestration for the Semantic Lexicon neural model."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .config import KnowledgeConfig, SemanticModelConfig
from .embeddings import GloVeEmbeddings
from .generator import (
    GenerationResult,
    PersonaGenerator,
    _maybe_generate_literal_response,
    _maybe_generate_structured_matrix_response,
)
from .intent import IntentClassifier, IntentExample
from .knowledge import KnowledgeEdge, KnowledgeNetwork
from .logging import configure_logging
from .persona import PersonaProfile, PersonaStore
from .template_learning import BalancedTutorPredictor
from .utils import seed_everything

LOGGER = configure_logging(logger_name=__name__)


@dataclass
class ModelArtifacts:
    embeddings_path: Optional[Path] = None
    intent_path: Optional[Path] = None
    knowledge_path: Optional[Path] = None


class NeuralSemanticModel:
    """Facade over model components."""

    def __init__(self, config: Optional[SemanticModelConfig] = None) -> None:
        self.config = config or SemanticModelConfig()
        seed_everything(0)
        self.embeddings = GloVeEmbeddings(self.config.embeddings)
        self.intent_classifier = IntentClassifier(self.config.intent)
        self.knowledge_network = KnowledgeNetwork(self.config.knowledge)
        self.persona_store = PersonaStore(self.config.persona)
        self.template_predictor = BalancedTutorPredictor.load_default()
        self.generator = PersonaGenerator(
            config=self.config.generator,
            embeddings=self.embeddings,
            knowledge=self.knowledge_network,
            template_predictor=self.template_predictor,
        )

    # Training --------------------------------------------------------------------
    def train_intents(self, examples: Iterable[IntentExample]) -> None:
        LOGGER.info("Training intent classifier")
        self.intent_classifier.fit(list(examples))

    def train_knowledge(self, edges: Iterable[KnowledgeEdge]) -> None:
        LOGGER.info("Training knowledge network")
        self.knowledge_network.fit(list(edges))

    # Inference -------------------------------------------------------------------
    def generate(self, prompt: str, persona: Optional[str] = None) -> GenerationResult:
        prompt_text = str(prompt or "")
        deterministic = _maybe_generate_structured_matrix_response(prompt_text)
        if deterministic is None:
            deterministic = _maybe_generate_literal_response(prompt_text)
        if deterministic is not None:
            intents = self._ranked_intents(prompt_text)
            return GenerationResult(
                response=deterministic,
                intents=intents,
                knowledge_hits=[],
                phrases=[],
                knowledge_selection=None,
            )
        intents = self._ranked_intents(prompt_text)
        profile = self.persona_store.get(persona)
        return self.generator.generate(prompt_text, profile, intents)

    def _ranked_intents(self, prompt: str, limit: int = 3) -> list[str]:
        try:
            intent_probs = self.intent_classifier.predict_proba(prompt)
        except Exception:  # pragma: no cover - defensive fallback
            LOGGER.debug(
                "Intent classifier unavailable; returning empty intent list",
                exc_info=True,
            )
            return []
        ranked = sorted(
            intent_probs,
            key=lambda label: intent_probs[label],
            reverse=True,
        )
        return ranked[:limit]

    # Persistence -----------------------------------------------------------------
    def save(self, directory: Path) -> ModelArtifacts:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        embeddings_path = directory / "embeddings.json"
        intent_path = directory / "intent.json"
        knowledge_path = directory / "knowledge.json"
        self.embeddings.save(embeddings_path)
        self._save_intent(intent_path)
        self._save_knowledge(knowledge_path)
        LOGGER.info("Saved model artifacts to %s", directory)
        return ModelArtifacts(
            embeddings_path=embeddings_path,
            intent_path=intent_path,
            knowledge_path=knowledge_path,
        )

    def _save_intent(self, path: Path) -> None:
        if self.intent_classifier.weights is None:
            raise ValueError("Intent classifier has not been trained")
        payload = {
            "weights": self.intent_classifier.weights.tolist(),
            "labels": self.intent_classifier.index_to_label,
            "vocabulary": self.intent_classifier.vocabulary,
        }
        with Path(path).open("w", encoding="utf8") as handle:
            json.dump(payload, handle)

    def _save_knowledge(self, path: Path) -> None:
        if (
            self.knowledge_network.embeddings is None
            or self.knowledge_network.relation_matrices is None
        ):
            raise ValueError("Knowledge network has not been trained")
        payload = {
            "entities": self.knowledge_network.entities,
            "relations": self.knowledge_network.relations,
            "embeddings": self.knowledge_network.embeddings.tolist(),
            "relation_matrices": (self.knowledge_network.relation_matrices.tolist()),
            "config": asdict(self.config.knowledge),
        }
        if self.knowledge_network.adjacency is not None:
            payload["adjacency"] = self.knowledge_network.adjacency.tolist()
        with Path(path).open("w", encoding="utf8") as handle:
            json.dump(payload, handle)

    # Loading ---------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        directory: Path,
        config: Optional[SemanticModelConfig] = None,
    ) -> NeuralSemanticModel:
        directory = Path(directory)
        instance = cls(config=config)
        instance.embeddings = GloVeEmbeddings.load(directory / "embeddings.json")
        instance._load_intent(directory / "intent.json")
        instance._load_knowledge(directory / "knowledge.json")
        instance.generator = PersonaGenerator(
            config=instance.config.generator,
            embeddings=instance.embeddings,
            knowledge=instance.knowledge_network,
            template_predictor=instance.template_predictor,
        )
        return instance

    def _load_intent(self, path: Path) -> None:
        with Path(path).open("r", encoding="utf8") as handle:
            payload = json.load(handle)
        vocabulary = {str(key): int(value) for key, value in payload["vocabulary"].items()}
        labels = {int(key): str(value) for key, value in payload["labels"].items()}
        self.intent_classifier.vocabulary = vocabulary
        self.intent_classifier.index_to_label = labels
        self.intent_classifier.label_to_index = {label: index for index, label in labels.items()}
        self.intent_classifier.weights = np.asarray(payload["weights"], dtype=float)
        self.intent_classifier._feature_indices = {
            feature: self.intent_classifier.vocabulary[feature]
            for feature in self.intent_classifier._feature_names
            if feature in self.intent_classifier.vocabulary
        }
        self.intent_classifier._finalise_weights()

    def _load_knowledge(self, path: Path) -> None:
        with Path(path).open("r", encoding="utf8") as handle:
            payload = json.load(handle)
        self.knowledge_network.entities = {
            str(key): int(value) for key, value in payload["entities"].items()
        }
        self.knowledge_network.relations = {
            str(key): int(value) for key, value in payload["relations"].items()
        }
        self.knowledge_network.embeddings = np.asarray(
            payload["embeddings"],
            dtype=float,
        )
        self.knowledge_network.relation_matrices = np.asarray(
            payload["relation_matrices"],
            dtype=float,
        )
        self.knowledge_network.config = KnowledgeConfig(**payload.get("config", {}))
        self.knowledge_network._build_index_lookup()
        adjacency = payload.get("adjacency")
        if adjacency is not None:
            self.knowledge_network.adjacency = np.asarray(adjacency, dtype=float)
            degrees = self.knowledge_network.adjacency.sum(axis=1)
            self.knowledge_network.degree = degrees
            laplacian = np.diag(degrees) - self.knowledge_network.adjacency
            self.knowledge_network.graph_laplacian = laplacian
            if np.all(degrees == 0):
                self.knowledge_network.transition = np.zeros_like(self.knowledge_network.adjacency)
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    transition = np.divide(
                        self.knowledge_network.adjacency,
                        degrees[:, None],
                        where=degrees[:, None] > 0,
                    )
                transition[np.isnan(transition)] = 0.0
                self.knowledge_network.transition = transition
        else:
            self.knowledge_network.adjacency = None
            self.knowledge_network.degree = None
            self.knowledge_network.graph_laplacian = None
            self.knowledge_network.transition = None
        self.knowledge_network.similarity = self.knowledge_network._compute_similarity_matrix()

    # Persona ---------------------------------------------------------------------
    def persona(self, name: Optional[str] = None) -> PersonaProfile:
        return self.persona_store.get(name)

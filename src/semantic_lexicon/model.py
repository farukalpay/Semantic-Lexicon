"""Top-level neural semantic model orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from .config import SemanticModelConfig
from .embeddings import GloVeEmbeddings
from .generator import GenerationRequest, PersonaGenerator
from .intent import IntentClassifier, IntentExample
from .knowledge import KnowledgeEntry, KnowledgeNetwork
from .logging import get_logger
from .persona import PersonaProfile, PersonaRegistry
from .utils import normalise_text, simple_tokenize, save_json
from .utils.random import seed_everything

LOGGER = get_logger(__name__)


@dataclass
class ModelArtifacts:
    embeddings_path: Optional[Path] = None
    intent_path: Optional[Path] = None
    knowledge_path: Optional[Path] = None


@dataclass
class NeuralSemanticModel:
    """Facade that orchestrates embeddings, intents, knowledge, and personas."""

    config: SemanticModelConfig = field(default_factory=SemanticModelConfig)
    embeddings: GloVeEmbeddings = field(default_factory=GloVeEmbeddings)
    intent: IntentClassifier = field(default_factory=IntentClassifier)
    knowledge: KnowledgeNetwork = field(default_factory=KnowledgeNetwork)
    personas: PersonaRegistry = field(default_factory=PersonaRegistry)

    def __post_init__(self) -> None:
        seed = seed_everything()
        LOGGER.debug("Initialised model with seed %d", seed)
        self.generator = PersonaGenerator(self.personas)
        self._ensure_default_personas()

    def _ensure_default_personas(self) -> None:
        for name, tone, temp in [
            ("researcher", "analytical", 0.6),
            ("tutor", "encouraging", 0.8),
            ("storyteller", "narrative", 0.9),
        ]:
            if name not in self.personas.personas:
                self.personas.add_persona(PersonaProfile(name=name, tone=tone, temperature=temp))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, workspace: Optional[Path] = None) -> ModelArtifacts:
        workspace = workspace or self.config.workspace
        workspace.mkdir(parents=True, exist_ok=True)
        embeddings_path = workspace / "embeddings.npz"
        intent_path = workspace / "intent.npz"
        knowledge_path = workspace / "knowledge.json"
        self.embeddings.save(embeddings_path)
        np.savez_compressed(intent_path, **self.intent.to_dict())
        from .utils import save_json

        save_json(knowledge_path, self.knowledge.to_dict())
        LOGGER.info("Persisted model artefacts to %s", workspace)
        return ModelArtifacts(
            embeddings_path=embeddings_path,
            intent_path=intent_path,
            knowledge_path=knowledge_path,
        )

    @classmethod
    def load(cls, artifacts: ModelArtifacts, config: Optional[SemanticModelConfig] = None) -> "NeuralSemanticModel":
        config = config or SemanticModelConfig()
        embeddings = GloVeEmbeddings.load_from_cache(artifacts.embeddings_path) if artifacts.embeddings_path else GloVeEmbeddings()
        intent_payload = np.load(artifacts.intent_path, allow_pickle=True) if artifacts.intent_path else None
        intent = IntentClassifier.from_dict({key: intent_payload[key] for key in intent_payload.files}) if intent_payload is not None else IntentClassifier()
        knowledge_payload = {}
        if artifacts.knowledge_path is not None and artifacts.knowledge_path.exists():
            with artifacts.knowledge_path.open("r", encoding="utf-8") as stream:
                knowledge_payload = json.load(stream)
        knowledge = KnowledgeNetwork.from_dict(knowledge_payload) if knowledge_payload else KnowledgeNetwork()
        model = cls(config=config, embeddings=embeddings, intent=intent, knowledge=knowledge)
        return model

    # ------------------------------------------------------------------
    # Inference utilities
    # ------------------------------------------------------------------
    def classify_intent(self, text: str) -> str:
        return self.intent.predict(text)

    def retrieve_concepts(self, text: str, *, top_k: int = 3) -> List[str]:
        tokens = simple_tokenize(text)
        candidates = [
            entry
            for entry in self.knowledge.nodes.values()
            if any(token in entry.description.lower() for token in tokens)
        ]
        concepts = [entry.concept for entry in candidates][:top_k]
        if len(concepts) < top_k:
            concepts.extend(list(self.knowledge.nodes.keys())[: top_k - len(concepts)])
        return concepts[:top_k]

    def generate_response(self, query: str, persona: Optional[str] = None, max_length: int = 60, temperature: Optional[float] = None) -> str:
        persona_name = persona or self.config.persona
        request = GenerationRequest(prompt=query, persona=persona_name, max_length=max_length, temperature=temperature or 0.8)
        return self.generator.generate(request)

    # ------------------------------------------------------------------
    # Training hooks
    # ------------------------------------------------------------------
    def ingest_corpus(self, corpus: Iterable[Dict[str, str]]) -> None:
        for record in corpus:
            concept = record.get("title") or record.get("concept")
            description = record.get("summary") or record.get("description") or ""
            related = record.get("related", [])
            if not concept:
                continue
            entry = KnowledgeEntry(concept=normalise_text(concept), description=description, related=[normalise_text(r) for r in related])
            self.knowledge.add_entry(entry)

    def train_intent(self, examples: Iterable[IntentExample], *, epochs: int = 3, lr: float = 0.1) -> None:
        self.intent.fit(list(examples), epochs=epochs, lr=lr)

    def build_from_corpus(self, corpus: Iterable[Dict[str, str]]) -> None:
        corpus_list = list(corpus)
        LOGGER.info("Building model artefacts from corpus of %d records", len(corpus_list))
        self.ingest_corpus(corpus_list)

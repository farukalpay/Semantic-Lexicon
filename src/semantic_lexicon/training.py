"""Training utilities for :mod:`semantic_lexicon`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple
from collections import Counter
import numpy as np

from .model import NeuralSemanticModel

TokenizeFn = Callable[[str], List[str]]
StopwordPredicate = Callable[[str], bool]


def default_tokenizer(text: str) -> List[str]:
    """A tiny whitespace tokenizer used when no tokenizer is supplied."""

    return [token for token in text.lower().strip().split() if token]


def default_stopword_predicate(token: str) -> bool:
    """Return ``True`` when *token* should be ignored."""

    return token in {"the", "and", "or", "is", "a", "an", "of"}


@dataclass
class CorpusProcessor:
    """Normalize raw corpus entries for downstream training."""

    tokenizer: TokenizeFn = default_tokenizer
    stopword_predicate: StopwordPredicate = default_stopword_predicate
    max_concepts: int = 10
    summary_window: int = 20
    vocabulary: Counter = field(default_factory=Counter)
    processed_entries: List[Dict[str, object]] = field(default_factory=list)

    def process_entry(self, entry: MutableMapping[str, object]) -> Dict[str, object]:
        """Return a normalized view of *entry* ready for model ingestion."""

        title = str(entry.get("title", ""))
        summary = str(entry.get("summary", ""))
        persona = str(entry.get("persona", "default"))
        source = str(entry.get("source", "unknown"))

        title_tokens = self.tokenizer(title)
        summary_tokens = self.tokenizer(summary)

        def keep(token: str) -> bool:
            return not self.stopword_predicate(token) and len(token) > 2

        concept_candidates = title_tokens + summary_tokens[: self.summary_window]
        concepts = [token for token in concept_candidates if keep(token)][: self.max_concepts]

        self.vocabulary.update(title_tokens)
        self.vocabulary.update(summary_tokens)

        normalized = {
            "title": title,
            "summary": summary,
            "title_tokens": title_tokens,
            "summary_tokens": summary_tokens,
            "concepts": concepts,
            "source": source,
            "persona": persona,
        }
        self.processed_entries.append(normalized)
        return normalized

    def prepare_corpus(self, entries: Iterable[MutableMapping[str, object]]) -> List[Dict[str, object]]:
        """Process each record in *entries* and return the cached list."""

        return [self.process_entry(entry) for entry in entries]


@dataclass
class NeuralTrainer:
    """Utility helpers around :class:`~semantic_lexicon.model.NeuralSemanticModel`."""

    model: NeuralSemanticModel
    training_history: Dict[str, List[float]] = field(
        default_factory=lambda: {"loss": [], "accuracy": [], "epochs": 0}
    )

    def create_training_pairs(self, processed_entries: Sequence[Dict[str, object]]) -> List[Tuple[str, str, str, Sequence[str]]]:
        """Synthesise extremely small training examples from *processed_entries*."""

        pairs: List[Tuple[str, str, str, Sequence[str]]] = []
        for entry in processed_entries:
            title = str(entry.get("title", ""))
            summary = str(entry.get("summary", ""))
            concepts = list(entry.get("concepts", []))

            if title:
                definition_response = summary[:200] if summary else f"{title} is a concept in the knowledge base."
                pairs.append((f"What is {title}?", definition_response, "definition", concepts))

            if len(concepts) >= 2:
                benefits = ", ".join(concepts[1:3]) if concepts[1:3] else "related concepts"
                benefit_response = f"{concepts[0]} helps with {benefits}."
                pairs.append((f"What are the benefits of {concepts[0]}?", benefit_response, "benefit", concepts))
        return pairs

    def train_intent_classifier(self, training_pairs: Sequence[Tuple[str, str, str, Sequence[str]]], epochs: int = 5) -> None:
        """Run a toy training loop over the intent classifier."""

        if not training_pairs:
            return

        intent_map = {intent: idx for idx, intent in enumerate(self.model.intent_classifier.INTENT_TYPES)}
        accuracy_history: List[float] = []

        for epoch in range(epochs):
            correct = 0
            total = 0
            for query, _, intent, _ in training_pairs:
                embeddings = self.model.embeddings.encode_sequence(query.lower().split())
                if embeddings.size == 0:
                    continue
                predicted_intent, _, _ = self.model.intent_classifier.classify(embeddings)
                total += 1
                if intent_map.get(predicted_intent, -1) == intent_map.get(intent, -1):
                    correct += 1
            accuracy = correct / total if total else 0.0
            accuracy_history.append(accuracy)
        self.training_history.setdefault("accuracy", []).extend(accuracy_history)
        self.training_history["epochs"] = epochs

    def train_knowledge_network(self, processed_entries: Sequence[Dict[str, object]]) -> None:
        """Populate the lightweight knowledge graph using *processed_entries*."""

        for entry in processed_entries:
            concepts = list(entry.get("concepts", []))
            summary_tokens = list(entry.get("summary_tokens", []))[:50]
            for concept in concepts:
                self.model.knowledge.add_concept(concept, summary_tokens)
            for idx, source_concept in enumerate(concepts):
                for target_concept in concepts[idx + 1 : idx + 3]:
                    if source_concept == target_concept:
                        continue
                    strength = 0.5 + 0.1 * min(3, concepts.count(source_concept) + concepts.count(target_concept))
                    self.model.knowledge.learn_relation(source_concept, target_concept, strength)

    def train_generator(self, training_pairs: Sequence[Tuple[str, str, str, Sequence[str]]], epochs: int = 3) -> None:
        """Simulate optimisation of the neural generator."""

        losses: List[float] = []
        for epoch in range(epochs):
            # The current implementation is illustrative only.
            losses.append(float(np.random.random() * 0.5 + 2.0 - (epoch * 0.3)))
        self.training_history.setdefault("loss", []).extend(losses)
        self.training_history["epochs"] = max(self.training_history.get("epochs", 0), epochs)


@dataclass
class LexiconMigrator:
    """Thin wrapper for comparing the neural model with legacy generators."""

    neural_model: NeuralSemanticModel

    def migrate_lexicon_query(self, prompt: str, persona: str = "default") -> str:
        return self.neural_model.generate_response(
            query=prompt,
            persona=persona,
            max_length=100,
            temperature=0.7,
        )

    def compare_outputs(self, prompt: str, old_lexicon_func: Optional[Callable[[str], str]] = None) -> Tuple[Optional[str], str]:
        old_output: Optional[str] = None
        if old_lexicon_func is not None:
            try:
                old_output = old_lexicon_func(prompt)
            except Exception as error:  # pragma: no cover - defensive fallback
                old_output = f"Old system error: {error}"
        new_output = self.migrate_lexicon_query(prompt)
        return old_output, new_output


__all__ = [
    "CorpusProcessor",
    "LexiconMigrator",
    "NeuralTrainer",
    "default_stopword_predicate",
    "default_tokenizer",
]

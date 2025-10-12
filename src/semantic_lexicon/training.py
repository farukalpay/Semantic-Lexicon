"""Training helpers for the Semantic Lexicon."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, MutableMapping, Sequence

from .config import TrainerConfig
from .intent import IntentExample
from .logging import get_logger
from .model import NeuralSemanticModel
from .utils import normalise_text, simple_tokenize

LOGGER = get_logger(__name__)

TokenizeFn = Callable[[str], List[str]]
StopwordPredicate = Callable[[str], bool]


def default_tokenizer(text: str) -> List[str]:
    return simple_tokenize(text)


def default_stopword_predicate(token: str) -> bool:
    return token in {"the", "and", "or", "is", "a", "an", "of"}


@dataclass
class CorpusProcessor:
    tokenizer: TokenizeFn = default_tokenizer
    stopword_predicate: StopwordPredicate = default_stopword_predicate
    max_concepts: int = 10
    summary_window: int = 40
    vocabulary: Counter = field(default_factory=Counter)
    processed_entries: List[Dict[str, object]] = field(default_factory=list)

    def process_entry(self, entry: MutableMapping[str, object]) -> Dict[str, object]:
        title = str(entry.get("title", ""))
        summary = str(entry.get("summary", ""))
        persona = str(entry.get("persona", "default"))
        tokens = self.tokenizer(summary)
        title_tokens = self.tokenizer(title)
        concepts = [token for token in title_tokens + tokens[: self.summary_window] if not self.stopword_predicate(token)]
        concepts = concepts[: self.max_concepts]
        self.vocabulary.update(tokens)
        normalised = {
            "title": title,
            "summary": summary,
            "persona": persona,
            "tokens": tokens,
            "concepts": concepts,
        }
        self.processed_entries.append(normalised)
        return normalised

    def prepare_corpus(self, entries: Iterable[MutableMapping[str, object]]) -> List[Dict[str, object]]:
        return [self.process_entry(entry) for entry in entries]


@dataclass
class TrainingArtifacts:
    intent_examples: List[IntentExample]
    knowledge_entries: List[Dict[str, object]]


@dataclass
class Trainer:
    model: NeuralSemanticModel
    config: TrainerConfig = field(default_factory=TrainerConfig)
    history: Dict[str, List[float]] = field(default_factory=lambda: {"loss": [], "accuracy": []})

    def prepare_training_data(self, processed_entries: Sequence[Dict[str, object]]) -> TrainingArtifacts:
        intent_examples: List[IntentExample] = []
        for entry in processed_entries:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            concepts = entry.get("concepts", [])
            if title:
                prompt = f"What is {title}?"
                intent_examples.append(IntentExample(text=prompt, label="definition"))
            if summary:
                prompt = f"How to apply {title}?"
                intent_examples.append(IntentExample(text=prompt, label="how_to"))
            if len(concepts) >= 2:
                prompt = f"Compare {concepts[0]} and {concepts[1]}"
                intent_examples.append(IntentExample(text=prompt, label="comparison"))
        knowledge_entries = [
            {
                "concept": normalise_text(entry.get("title", "")),
                "description": entry.get("summary", ""),
                "related": [normalise_text(concept) for concept in entry.get("concepts", [])],
            }
            for entry in processed_entries
            if entry.get("title")
        ]
        return TrainingArtifacts(intent_examples=intent_examples, knowledge_entries=knowledge_entries)

    def train(self, processed_entries: Sequence[Dict[str, object]]) -> None:
        artifacts = self.prepare_training_data(processed_entries)
        self.model.ingest_corpus(artifacts.knowledge_entries)
        self.model.train_intent(artifacts.intent_examples, epochs=self.config.epochs, lr=self.config.learning_rate)
        LOGGER.info("Training complete with %d intent examples", len(artifacts.intent_examples))

    def save_training_report(self, path: Path) -> None:
        payload = {"history": self.history}
        from .utils import save_json

        save_json(path, payload)
        LOGGER.info("Wrote training report to %s", path)


__all__ = ["CorpusProcessor", "Trainer", "TrainingArtifacts", "default_tokenizer", "default_stopword_predicate"]

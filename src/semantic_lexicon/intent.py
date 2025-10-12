"""Intent classification components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .logging import get_logger
from .utils.text import simple_tokenize

LOGGER = get_logger(__name__)

INTENTS = [
    "definition",
    "comparison",
    "how_to",
    "benefit",
    "identity",
    "general",
]


@dataclass
class IntentExample:
    text: str
    label: str


@dataclass
class IntentClassifier:
    """Deterministic bag-of-words intent classifier."""

    labels: Sequence[str] = field(default_factory=lambda: INTENTS)
    embedding_dim: int = 50

    def __post_init__(self) -> None:
        self.label_to_index = {label: i for i, label in enumerate(self.labels)}
        self.weights = np.zeros((len(self.labels), self.embedding_dim), dtype="float32")
        self.bias = np.zeros(len(self.labels), dtype="float32")

    def featurise(self, tokens: Iterable[str]) -> np.ndarray:
        """Represent ``tokens`` as mean pooled indicator vector."""
        counts = np.zeros(self.embedding_dim, dtype="float32")
        for token in tokens:
            idx = hash(token) % self.embedding_dim
            counts[idx] += 1
        if counts.sum() > 0:
            counts /= counts.sum()
        return counts

    def predict_proba(self, text: str) -> np.ndarray:
        tokens = simple_tokenize(text)
        features = self.featurise(tokens)
        logits = self.weights @ features + self.bias
        exp = np.exp(logits - logits.max())
        return exp / exp.sum()

    def predict(self, text: str) -> str:
        proba = self.predict_proba(text)
        return self.labels[int(np.argmax(proba))]

    def fit(self, examples: Sequence[IntentExample], *, epochs: int = 5, lr: float = 0.1) -> None:
        """Train using a perceptron-style update."""
        if not examples:
            LOGGER.warning("No intent examples provided; skipping training")
            return
        for epoch in range(epochs):
            total_loss = 0.0
            for example in examples:
                features = self.featurise(simple_tokenize(example.text))
                logits = self.weights @ features + self.bias
                exp = np.exp(logits - logits.max())
                proba = exp / exp.sum()
                target = np.zeros(len(self.labels), dtype="float32")
                target[self.label_to_index[example.label]] = 1.0
                error = proba - target
                self.weights -= lr * np.outer(error, features)
                self.bias -= lr * error
                total_loss += float(np.dot(error, error))
            LOGGER.debug("Intent epoch %d loss %.4f", epoch + 1, total_loss / max(len(examples), 1))

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {"weights": self.weights, "bias": self.bias, "labels": np.array(self.labels)}

    @classmethod
    def from_dict(cls, payload: Dict[str, np.ndarray]) -> "IntentClassifier":
        labels = [str(label) for label in payload["labels"]]
        instance = cls(labels=labels, embedding_dim=payload["weights"].shape[1])
        instance.weights = payload["weights"]
        instance.bias = payload["bias"]
        return instance

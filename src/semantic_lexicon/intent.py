"""Intent classification module."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import IntentConfig
from .logging import configure_logging
from .utils import tokenize

LOGGER = configure_logging(logger_name=__name__)


@dataclass(frozen=True)
class IntentExample:
    """Training example for the intent classifier."""

    text: str
    intent: str


class IntentClassifier:
    """A simple NumPy-based multinomial logistic regression model."""

    def __init__(self, config: IntentConfig | None = None) -> None:
        self.config = config or IntentConfig()
        self.label_to_index: dict[str, int] = {}
        self.index_to_label: dict[int, str] = {}
        self.vocabulary: dict[str, int] = {}
        self.weights: NDArray[np.float64] | None = None

    # Training --------------------------------------------------------------------
    def fit(self, examples: Iterable[IntentExample]) -> None:
        dataset = list(examples)
        if not dataset:
            raise ValueError("No examples supplied")
        self._prepare_labels(dataset)
        matrix = self._vectorise(dataset)
        labels = np.array([self.label_to_index[item.intent] for item in dataset], dtype=int)
        num_features = matrix.shape[1]
        num_labels = len(self.label_to_index)
        rng = np.random.default_rng(0)
        self.weights = np.asarray(
            rng.normal(0, 0.1, size=(num_features, num_labels)),
            dtype=float,
        )
        for epoch in range(self.config.epochs):
            logits = np.asarray(matrix @ self.weights, dtype=float)
            probs = self._softmax(logits)
            one_hot = np.eye(num_labels, dtype=float)[labels]
            gradient = matrix.T @ (probs - one_hot) / len(dataset)
            self.weights -= self.config.learning_rate * gradient
            loss = -np.mean(np.log(probs[np.arange(len(dataset)), labels] + 1e-12))
            LOGGER.debug("Intent epoch %s | loss=%.4f", epoch + 1, loss)
        LOGGER.info("Trained intent classifier with %d intents", num_labels)

    def _prepare_labels(self, examples: Sequence[IntentExample]) -> None:
        for example in examples:
            if example.intent not in self.label_to_index:
                index = len(self.label_to_index)
                self.label_to_index[example.intent] = index
                self.index_to_label[index] = example.intent

    def _build_vocabulary(self, texts: Sequence[Sequence[str]]) -> None:
        vocab = sorted({token for tokens in texts for token in tokens})
        self.vocabulary = {token: idx for idx, token in enumerate(vocab)}
        LOGGER.debug("Built vocabulary of size %d", len(self.vocabulary))

    def _vectorise(self, examples: Sequence[IntentExample]) -> NDArray[np.float64]:
        tokenised = [tokenize(example.text) for example in examples]
        if not self.vocabulary:
            self._build_vocabulary(tokenised)
        matrix = np.zeros((len(tokenised), len(self.vocabulary)), dtype=float)
        for row, tokens in enumerate(tokenised):
            for token in tokens:
                if token in self.vocabulary:
                    matrix[row, self.vocabulary[token]] += 1.0
        return matrix

    # Prediction ------------------------------------------------------------------
    def predict(self, text: str) -> str:
        probabilities = self.predict_proba(text)
        return max(probabilities, key=lambda label: probabilities[label])

    def predict_proba(self, text: str) -> dict[str, float]:
        if self.weights is None:
            raise ValueError("Classifier has not been trained")
        vector = np.zeros((len(self.vocabulary),), dtype=float)
        for token in tokenize(text):
            if token in self.vocabulary:
                vector[self.vocabulary[token]] += 1.0
        logits = vector @ self.weights
        probs = self._softmax(logits[np.newaxis, :])[0]
        return {self.index_to_label[i]: float(prob) for i, prob in enumerate(probs)}

    @staticmethod
    def _softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        denominator = np.sum(exp, axis=1, keepdims=True)
        return np.asarray(exp / denominator, dtype=float)

"""Learned predictors for templated responses."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .templates import BalancedTutorTemplate
from .utils import read_jsonl, tokenize

__all__ = [
    "BalancedTutorExample",
    "BalancedTutorPredictor",
    "load_balanced_tutor_dataset",
]

_STOP_TOKEN = "<STOP>"


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BalancedTutorExample:
    """Training example describing template variables for a prompt."""

    prompt: str
    intent: str
    topics: Sequence[str]
    actions: Sequence[str]

    def __post_init__(self) -> None:
        if len(self.topics) != len(self.actions):
            raise ValueError("topics and actions must be the same length")
        if not self.topics:
            raise ValueError("at least one topic/action pair is required")

    def as_template(self) -> BalancedTutorTemplate:
        """Convert the example into a :class:`BalancedTutorTemplate`."""

        return BalancedTutorTemplate(
            prompt=self.prompt,
            intent=self.intent,
            topics=tuple(self.topics),
            actions=tuple(self.actions),
        )


class _SoftmaxModel:
    """Simple multi-class logistic regression trained with gradient descent."""

    def __init__(
        self,
        classes: Iterable[str],
        n_features: int,
        *,
        learning_rate: float = 0.4,
        epochs: int = 400,
        l2: float = 1e-3,
        loss_weight: float = 1.0,
    ) -> None:
        self.classes = tuple(dict.fromkeys(classes))
        if not self.classes:
            raise ValueError("at least one class is required")
        self.n_features = int(n_features)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.l2 = float(l2)
        self.loss_weight = float(loss_weight)
        self._constant_class: str | None = None
        self._weights: np.ndarray | None = None
        self._bias: np.ndarray | None = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: Sequence[str]) -> None:
        y = list(y)
        if len(set(y)) == 1:
            self._constant_class = y[0]
            self._weights = None
            self._bias = None
            return

        class_to_index = {label: index for index, label in enumerate(self.classes)}
        y_indices = np.array([class_to_index[label] for label in y], dtype=int)
        n_samples = X.shape[0]
        weights = np.zeros((len(self.classes), self.n_features), dtype=float)
        bias = np.zeros(len(self.classes), dtype=float)

        for _ in range(self.epochs):
            logits = X @ weights.T + bias
            probs = _softmax(logits)
            probs[np.arange(n_samples), y_indices] -= 1.0
            probs /= n_samples

            grad_w = probs.T @ X + self.l2 * weights
            grad_b = probs.sum(axis=0)

            step = self.learning_rate * self.loss_weight
            weights -= step * grad_w
            bias -= step * grad_b

        self._weights = weights
        self._bias = bias

    # ------------------------------------------------------------------
    def predict(self, vector: np.ndarray) -> str:
        if self._constant_class is not None:
            return self._constant_class
        probs = self.probabilities(vector)
        index = int(np.argmax(probs))
        return self.classes[index]

    def predict_topk(self, vector: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        if self._constant_class is not None:
            return [(self._constant_class, 1.0)]
        probs = self.probabilities(vector)
        order = np.argsort(probs)[::-1][:k]
        return [(self.classes[idx], float(probs[idx])) for idx in order]

    def probabilities(self, vector: np.ndarray) -> np.ndarray:
        if self._constant_class is not None:
            probs = np.zeros(len(self.classes), dtype=float)
            probs[0] = 1.0
            return probs
        logits = vector @ self._weights.T + self._bias  # type: ignore[union-attr]
        return _softmax(logits)


def _softmax(logits: np.ndarray) -> FloatArray:
    if logits.ndim == 1:
        logits = logits[None, :]
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    denom = exp.sum(axis=-1, keepdims=True)
    probs = cast(FloatArray, exp / np.maximum(denom, 1e-12))
    if probs.shape[0] == 1:
        return cast(FloatArray, probs.squeeze(axis=0))
    return probs


class BalancedTutorPredictor:
    """Learns a mapping from prompts to template variables."""

    def __init__(
        self,
        examples: Sequence[BalancedTutorExample],
        *,
        lambda_intent: float = 1.0,
        lambda_topics: float = 1.0,
        lambda_actions: float = 1.0,
    ) -> None:
        if not examples:
            raise ValueError("at least one example is required")
        self.examples = list(examples)
        self.lambda_intent = float(lambda_intent)
        self.lambda_topics = float(lambda_topics)
        self.lambda_actions = float(lambda_actions)

        self._intent_defaults: dict[str, BalancedTutorExample] = {}
        for example in self.examples:
            self._intent_defaults.setdefault(example.intent, example)

        self._token_to_topics = _build_topic_index(self.examples)
        self._vocabulary = self._build_vocabulary(self.examples)
        self._token_to_index = {token: index for index, token in enumerate(self._vocabulary)}
        self._idf = self._compute_idf(self.examples)

        self._feature_matrix = cast(
            FloatArray,
            np.vstack([self._vectorise(example.prompt) for example in self.examples]),
        )
        self._train_models()

    # ------------------------------------------------------------------
    @classmethod
    def from_jsonl(cls, path: str | Path, **kwargs: float) -> BalancedTutorPredictor:
        """Load examples from a JSONL file and create a predictor."""

        examples = load_balanced_tutor_dataset(path)
        return cls(examples, **kwargs)

    @classmethod
    def load_default(cls, **kwargs: float) -> BalancedTutorPredictor:
        """Load the bundled training set for balanced tutor prompts."""

        data_path = Path(__file__).resolve().parent / "data" / "balanced_tutor_training.jsonl"
        return cls.from_jsonl(data_path, **kwargs)

    # ------------------------------------------------------------------
    def predict_variables(self, prompt: str) -> BalancedTutorExample:
        """Predict the intent/topics/actions tuple for ``prompt``."""

        vector = self._vectorise(prompt)
        intent = self._predict_intent(vector)
        topics, actions = self._predict_topics_and_actions(vector, intent, prompt)
        if not topics or not actions:
            fallback = self._intent_defaults.get(intent) or self.examples[0]
            topics = list(fallback.topics)
            actions = list(fallback.actions)
        limit = min(len(topics), len(actions))
        topics = topics[:limit]
        actions = actions[:limit]
        if not topics:
            fallback = self.examples[0]
            topics = list(fallback.topics)
            actions = list(fallback.actions)
        return BalancedTutorExample(
            prompt=prompt,
            intent=intent,
            topics=tuple(topics),
            actions=tuple(actions),
        )

    def predict(self, prompt: str) -> BalancedTutorTemplate:
        """Predict a :class:`BalancedTutorTemplate` for ``prompt``."""

        example = self.predict_variables(prompt)
        return BalancedTutorTemplate(
            prompt=prompt,
            intent=example.intent,
            topics=tuple(example.topics),
            actions=tuple(example.actions),
        )

    # ------------------------------------------------------------------
    def _train_models(self) -> None:
        intents = [example.intent for example in self.examples]
        max_topics = max(len(example.topics) for example in self.examples)
        max_actions = max(len(example.actions) for example in self.examples)
        max_slots = max(max_topics, max_actions)

        self._intent_model = _SoftmaxModel(
            intents,
            self._feature_matrix.shape[1],
            loss_weight=self.lambda_intent,
        )
        self._intent_model.fit(self._feature_matrix, intents)

        self._topic_models: list[_SoftmaxModel] = []
        self._action_models: list[_SoftmaxModel] = []

        for slot in range(max_slots):
            topic_labels = [
                example.topics[slot] if slot < len(example.topics) else _STOP_TOKEN
                for example in self.examples
            ]
            action_labels = [
                example.actions[slot] if slot < len(example.actions) else _STOP_TOKEN
                for example in self.examples
            ]

            topic_model = _SoftmaxModel(
                topic_labels,
                self._feature_matrix.shape[1],
                loss_weight=self.lambda_topics / max(1, max_slots),
            )
            topic_model.fit(self._feature_matrix, topic_labels)
            self._topic_models.append(topic_model)

            action_model = _SoftmaxModel(
                action_labels,
                self._feature_matrix.shape[1],
                loss_weight=self.lambda_actions / max(1, max_slots),
            )
            action_model.fit(self._feature_matrix, action_labels)
            self._action_models.append(action_model)

        self._actions_by_intent = _build_actions_by_intent(self.examples)

    def _predict_intent(self, vector: np.ndarray) -> str:
        return self._intent_model.predict(vector)

    def _predict_topics_and_actions(
        self, vector: np.ndarray, intent: str, prompt: str
    ) -> tuple[list[str], list[str]]:
        topics: list[str] = []
        actions: list[str] = []

        keyword_votes = _keyword_votes(prompt, self._token_to_topics)
        allowed_actions = self._actions_by_intent.get(intent, set())

        for topic_model, action_model in zip(self._topic_models, self._action_models):
            topic_candidates = topic_model.predict_topk(vector, k=3)
            topic = _select_topic(topic_candidates, keyword_votes)
            if topic == _STOP_TOKEN:
                break

            action = _select_action(action_model, vector, allowed_actions)
            if action == _STOP_TOKEN:
                break

            topics.append(topic)
            actions.append(action)
        return topics, actions

    # ------------------------------------------------------------------
    def _build_vocabulary(self, examples: Sequence[BalancedTutorExample]) -> list[str]:
        tokens: set[str] = set()
        for example in examples:
            tokens.update(tokenize(example.prompt))
        return sorted(tokens)

    def _compute_idf(self, examples: Sequence[BalancedTutorExample]) -> FloatArray:
        document_count = len(examples)
        idf: FloatArray = np.zeros(len(self._vocabulary), dtype=float)
        for example in examples:
            seen_tokens = set(tokenize(example.prompt))
            for token in seen_tokens:
                if token in self._token_to_index:
                    idf[self._token_to_index[token]] += 1.0
        idf = cast(FloatArray, np.log((1.0 + document_count) / (1.0 + idf)) + 1.0)
        return idf

    def _vectorise(self, prompt: str) -> FloatArray:
        counts = Counter(tokenize(prompt))
        vector: FloatArray = np.zeros(len(self._vocabulary), dtype=float)
        if not counts:
            return vector
        total = sum(counts.values())
        for token, count in counts.items():
            index = self._token_to_index.get(token)
            if index is None:
                continue
            vector[index] = (count / total) * self._idf[index]
        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm
        return vector


def _keyword_votes(prompt: str, token_to_topics: dict[str, Counter]) -> Counter:
    votes: Counter = Counter()
    for token in tokenize(prompt):
        for topic, count in token_to_topics.get(token, {}).items():
            votes[topic] += count
    return votes


def _select_topic(
    candidates: list[tuple[str, float]],
    keyword_votes: Counter,
    *,
    keyword_weight: float = 0.15,
) -> str:
    best_topic = candidates[0][0]
    best_score = -np.inf
    for topic, probability in candidates:
        score = probability
        if keyword_votes:
            score += keyword_weight * keyword_votes.get(topic, 0)
        if score > best_score:
            best_score = score
            best_topic = topic
    return best_topic


def _select_action(
    model: _SoftmaxModel,
    vector: np.ndarray,
    allowed_actions: set[str],
) -> str:
    candidates = model.predict_topk(vector, k=len(model.classes))
    if not allowed_actions:
        return candidates[0][0]
    best_action = candidates[0][0]
    best_score = -np.inf
    for action, score in candidates:
        if action not in allowed_actions:
            continue
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def _build_topic_index(examples: Sequence[BalancedTutorExample]) -> dict[str, Counter]:
    mapping: dict[str, Counter] = defaultdict(Counter)
    for example in examples:
        for topic in example.topics:
            for token in tokenize(topic):
                mapping[token][topic] += 1
    return mapping


def _build_actions_by_intent(examples: Sequence[BalancedTutorExample]) -> dict[str, set[str]]:
    actions: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        actions[example.intent].update(example.actions)
    return actions


def load_balanced_tutor_dataset(path: str | Path) -> list[BalancedTutorExample]:
    """Load balanced tutor examples from ``path``."""

    resolved = Path(path)
    return [
        BalancedTutorExample(
            prompt=str(payload["prompt"]),
            intent=str(payload["intent"]),
            topics=tuple(str(value) for value in _expect_sequence(payload["topics"])),
            actions=tuple(str(value) for value in _expect_sequence(payload["actions"])),
        )
        for payload in read_jsonl(resolved)
    ]


def _expect_sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    raise TypeError(f"Expected a sequence, received {type(value)!r}")

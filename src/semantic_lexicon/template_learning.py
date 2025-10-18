# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""Learned predictors for templated responses."""

from __future__ import annotations

import logging
import math
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

LOGGER = logging.getLogger(__name__)

_STOP_TOKEN = "<STOP>"


FloatArray = NDArray[np.float64]


_MAX_FEATURE_MAGNITUDE = 1024.0
_MAX_PARAMETER_MAGNITUDE = 64.0
_MAX_LOGIT_MAGNITUDE = 60.0
_MAX_PROBABILITY_MAGNITUDE = 1.0
_STEP_SCALE_THRESHOLD = 512.0
_NORMALISATION_EPS = 1e-8


def _clamp_inplace(array: np.ndarray, limit: float) -> np.ndarray:
    """Clamp values in ``array`` to ``[-limit, limit]`` after sanitising NaNs."""

    arr = np.asarray(array, dtype=np.float64)
    if arr.size == 0:
        return arr
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=limit, neginf=-limit)
    if limit > 0.0:
        np.clip(arr, -limit, limit, out=arr)
    return arr


def _safe_matmul(
    left: np.ndarray,
    right: np.ndarray,
    *,
    context: str,
    left_limit: float,
    right_limit: float,
) -> FloatArray:
    """Compute ``left @ right`` with operand clamping and NaN sanitisation."""

    left_local = np.asarray(left, dtype=np.float64)
    right_local = np.asarray(right, dtype=np.float64)
    if left_local.ndim == 0 or right_local.ndim == 0:
        return cast(FloatArray, np.zeros((), dtype=np.float64))
    left_local = left_local.copy()
    right_local = right_local.copy()
    _clamp_inplace(left_local, left_limit)
    _clamp_inplace(right_local, right_limit)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        product = left_local @ right_local
    product_array = np.asarray(product, dtype=np.float64)
    if not np.isfinite(product_array).all():
        LOGGER.warning("%s produced non-finite values; applying sanitisation", context)
        product_array = np.nan_to_num(product_array, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return cast(FloatArray, product_array)


def _clip_gradients(
    grad_w: np.ndarray, grad_b: np.ndarray, clip_norm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Clip gradients to the provided ``clip_norm`` when necessary."""

    clip_norm = float(max(clip_norm, 0.0))
    if clip_norm == 0.0:
        return grad_w, grad_b
    total_norm = float(
        math.sqrt(np.sum(grad_w**2, dtype=np.float64) + np.sum(grad_b**2, dtype=np.float64))
    )
    if not math.isfinite(total_norm) or total_norm <= clip_norm or total_norm == 0.0:
        return grad_w, grad_b
    scale = clip_norm / (total_norm + _NORMALISATION_EPS)
    return grad_w * scale, grad_b * scale


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
        learning_rate: float = 0.05,
        epochs: int = 400,
        l2: float = 1e-3,
        loss_weight: float = 1.0,
        gradient_clip_norm: float = 1.0,
    ) -> None:
        self.classes = tuple(dict.fromkeys(classes))
        if not self.classes:
            raise ValueError("at least one class is required")
        self.n_features = int(n_features)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.l2 = float(l2)
        self.loss_weight = float(loss_weight)
        self.gradient_clip_norm = float(max(gradient_clip_norm, 0.0))
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
        features = _normalise_design_matrix(X)
        if not _is_finite(features):
            LOGGER.warning("Softmax training matrix contained non-finite values; re-normalising")
            features = _normalise_design_matrix(features)
        if not _is_finite(features):
            LOGGER.warning(
                "Softmax features still non-finite after normalisation; zeroing offending rows"
            )
            features = np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        _clamp_inplace(features, _MAX_FEATURE_MAGNITUDE)
        n_samples = features.shape[0]
        feature_count = max(self.n_features, 1)
        class_count = max(len(self.classes), 1)
        rng = np.random.default_rng(0)
        limit = math.sqrt(6.0 / (feature_count + class_count))
        weights = rng.uniform(-limit, limit, size=(class_count, self.n_features)).astype(float)
        bias = np.zeros(len(self.classes), dtype=float)
        _clamp_inplace(weights, _MAX_PARAMETER_MAGNITUDE)
        _clamp_inplace(bias, _MAX_PARAMETER_MAGNITUDE)

        base_step = self.learning_rate * self.loss_weight
        step_scale = max(_safe_step_scale(features), 1.0)
        scale_factor = max(step_scale / _STEP_SCALE_THRESHOLD, 1.0)
        step = base_step / math.sqrt(scale_factor)
        if step_scale > _STEP_SCALE_THRESHOLD:
            damping = math.log(step_scale / _STEP_SCALE_THRESHOLD + 1.0)
            damping = min(max(damping, 1.0), 1.5)
            step /= damping
        if step == 0.0 and base_step > 0.0:
            step = base_step * 1e-6
        min_step = step * 1e-3 if step else 0.0

        dataset_size = max(n_samples, 1)
        for _ in range(self.epochs):
            _clamp_inplace(weights, _MAX_PARAMETER_MAGNITUDE)
            _clamp_inplace(bias, _MAX_PARAMETER_MAGNITUDE)
            logits = _safe_matmul(
                features,
                weights.T,
                context="Softmax logits",
                left_limit=_MAX_FEATURE_MAGNITUDE,
                right_limit=_MAX_PARAMETER_MAGNITUDE,
            )
            logits += bias
            np.clip(logits, -_MAX_LOGIT_MAGNITUDE, _MAX_LOGIT_MAGNITUDE, out=logits)
            probs = _softmax(logits)
            probs[np.arange(n_samples), y_indices] -= 1.0
            probs /= dataset_size
            probs = np.nan_to_num(probs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            grad_w = _safe_matmul(
                probs.T,
                features,
                context="Softmax weight gradient",
                left_limit=_MAX_PROBABILITY_MAGNITUDE,
                right_limit=_MAX_FEATURE_MAGNITUDE,
            )
            grad_w += self.l2 * weights
            grad_b = probs.sum(axis=0)
            grad_w = np.nan_to_num(grad_w, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            grad_b = np.nan_to_num(grad_b, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            grad_w, grad_b = _clip_gradients(grad_w, grad_b, self.gradient_clip_norm)

            update_w = step * grad_w
            update_b = step * grad_b

            new_weights = weights - update_w
            new_bias = bias - update_b

            if not _is_finite(new_weights) or not _is_finite(new_bias):
                if step <= min_step or step == 0.0:
                    LOGGER.warning("Softmax training aborted due to non-finite updates")
                    break
                step *= 0.5
                continue

            weights = np.nan_to_num(new_weights, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            bias = np.nan_to_num(new_bias, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            _clamp_inplace(weights, _MAX_PARAMETER_MAGNITUDE)
            _clamp_inplace(bias, _MAX_PARAMETER_MAGNITUDE)

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
        vector64 = _clamp_inplace(np.asarray(vector, dtype=np.float64), _MAX_FEATURE_MAGNITUDE)
        weights = _clamp_inplace(self._weights, _MAX_PARAMETER_MAGNITUDE)  # type: ignore[arg-type]
        bias = _clamp_inplace(self._bias, _MAX_PARAMETER_MAGNITUDE)  # type: ignore[arg-type]
        logits = _safe_matmul(
            vector64,
            weights.T,  # type: ignore[union-attr]
            context="Softmax logits (inference)",
            left_limit=_MAX_FEATURE_MAGNITUDE,
            right_limit=_MAX_PARAMETER_MAGNITUDE,
        )
        logits += bias  # type: ignore[union-attr]
        np.clip(logits, -_MAX_LOGIT_MAGNITUDE, _MAX_LOGIT_MAGNITUDE, out=logits)
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


def _safe_step_scale(matrix: np.ndarray) -> float:
    """Return a scaling factor that keeps gradient steps numerically stable."""

    if matrix.size == 0:
        return 1.0
    try:
        spectral_norm = float(np.linalg.norm(matrix, ord=2))
    except ValueError:  # pragma: no cover - defensive: ord=2 unsupported
        spectral_norm = float(np.linalg.norm(matrix))
    if not np.isfinite(spectral_norm) or spectral_norm == 0.0:
        return 1.0
    scaled = spectral_norm * spectral_norm
    if not np.isfinite(scaled) or scaled == 0.0:
        return 1.0
    return scaled


def _normalise_design_matrix(matrix: np.ndarray) -> FloatArray:
    r"""Return a finite matrix with bounded absolute feature values."""

    if matrix.size == 0:
        return cast(FloatArray, np.asarray(matrix, dtype=np.float64))
    normalised = np.asarray(matrix, dtype=np.float64)
    if normalised.ndim == 1:
        return _normalise_feature_vector(normalised)
    np.nan_to_num(normalised, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    column_mean = normalised.mean(axis=0, keepdims=True)
    normalised -= column_mean
    column_var = normalised.var(axis=0, keepdims=True)
    column_std = np.sqrt(column_var)
    column_std = np.where(column_std < _NORMALISATION_EPS, 1.0, column_std)
    normalised /= column_std
    np.nan_to_num(normalised, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    row_max = np.max(np.abs(normalised), axis=1, keepdims=True, initial=0.0)
    large_rows = row_max > _MAX_FEATURE_MAGNITUDE
    if np.any(large_rows):
        normalised[large_rows] *= _MAX_FEATURE_MAGNITUDE / row_max[large_rows]
    max_abs = float(np.max(np.abs(normalised)))
    if not np.isfinite(max_abs) or max_abs <= _MAX_FEATURE_MAGNITUDE or max_abs == 0.0:
        return cast(FloatArray, normalised)
    scale = _MAX_FEATURE_MAGNITUDE / max_abs
    return cast(FloatArray, normalised * scale)


def _normalise_feature_vector(vector: np.ndarray) -> FloatArray:
    r"""Return a finite vector with bounded absolute feature values."""

    if vector.size == 0:
        return cast(FloatArray, np.asarray(vector, dtype=np.float64))
    normalised = np.asarray(vector, dtype=np.float64)
    np.nan_to_num(normalised, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    mean = float(normalised.mean()) if normalised.size else 0.0
    normalised -= mean
    std = float(normalised.std()) if normalised.size else 1.0
    if not math.isfinite(std) or std < _NORMALISATION_EPS:
        std = 1.0
    normalised /= std
    np.nan_to_num(normalised, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    max_abs = float(np.max(np.abs(normalised)))
    if not np.isfinite(max_abs) or max_abs <= _MAX_FEATURE_MAGNITUDE or max_abs == 0.0:
        return cast(FloatArray, normalised)
    scale = _MAX_FEATURE_MAGNITUDE / max_abs
    return cast(FloatArray, normalised * scale)


def _is_finite(array: np.ndarray) -> bool:
    """Return ``True`` when all elements of ``array`` are finite."""

    return bool(np.isfinite(array).all())


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
            _normalise_design_matrix(
                np.vstack([self._vectorise(example.prompt) for example in self.examples])
            ),
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
        return _normalise_feature_vector(vector)


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

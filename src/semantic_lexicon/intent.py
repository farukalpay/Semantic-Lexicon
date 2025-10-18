# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""Intent classification module."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Optional

import math

import numpy as np
from numpy.typing import NDArray

from .analysis import (
    DirichletCalibrator,
    RewardComponents,
    composite_reward,
    confidence_reward,
    correctness_reward,
    estimate_optimal_weights,
    feedback_reward,
    project_to_simplex,
    semantic_similarity_reward,
)
from .analysis.error import compute_confusion_correction
from .config import IntentConfig
from .logging import configure_logging
from .utils import normalise_text, tokenize

LOGGER = configure_logging(logger_name=__name__)


@dataclass(frozen=True)
class IntentExample:
    """Training example for the intent classifier."""

    text: str
    intent: str
    feedback: float = 0.95


_MAX_FEATURE_MAGNITUDE = 1024.0
_STEP_SCALE_THRESHOLD = 512.0


def _safe_step_scale(matrix: np.ndarray) -> float:
    """Return a scaling factor that dampens gradient steps for stability."""

    if matrix.size == 0:
        return 1.0
    try:
        spectral_norm = float(np.linalg.norm(matrix, ord=2))
    except ValueError:  # pragma: no cover - defensive
        spectral_norm = float(np.linalg.norm(matrix))
    if not np.isfinite(spectral_norm) or spectral_norm == 0.0:
        return 1.0
    scaled = spectral_norm * spectral_norm
    if not np.isfinite(scaled) or scaled == 0.0:
        return 1.0
    return scaled


def _normalise_design_matrix(matrix: np.ndarray) -> np.ndarray:
    r"""Return a finite matrix with bounded absolute feature values."""

    if matrix.size == 0:
        return np.asarray(matrix, dtype=np.float64)
    normalised = np.asarray(matrix, dtype=np.float64)
    np.nan_to_num(normalised, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    max_abs = float(np.max(np.abs(normalised)))
    if not np.isfinite(max_abs) or max_abs <= _MAX_FEATURE_MAGNITUDE or max_abs == 0.0:
        return normalised
    scale = _MAX_FEATURE_MAGNITUDE / max_abs
    return normalised * scale


def _normalise_feature_vector(vector: np.ndarray) -> np.ndarray:
    r"""Return a finite feature vector with bounded absolute values."""

    if vector.size == 0:
        return np.asarray(vector, dtype=np.float64)
    normalised = np.asarray(vector, dtype=np.float64)
    np.nan_to_num(normalised, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if normalised.size == 0:
        return normalised
    max_abs = float(np.max(np.abs(normalised)))
    if not np.isfinite(max_abs) or max_abs <= _MAX_FEATURE_MAGNITUDE or max_abs == 0.0:
        return normalised
    scale = _MAX_FEATURE_MAGNITUDE / max_abs
    return normalised * scale


class IntentClassifier:
    """A simple NumPy-based multinomial logistic regression model."""

    def __init__(self, config: Optional[IntentConfig] = None) -> None:
        self.config = config or IntentConfig()
        self.label_to_index: dict[str, int] = {}
        self.index_to_label: dict[int, str] = {}
        self.vocabulary: dict[str, int] = {}
        self.weights: Optional[np.ndarray] = None
        self._weights_for_inference: Optional[NDArray[np.float64]] = None
        self._calibrator: Optional[DirichletCalibrator] = None
        self._correction_matrix: Optional[NDArray[np.float64]] = None
        self._posterior_predictive: Optional[NDArray[np.float64]] = None
        self._reward_weights: Optional[NDArray[np.float64]] = None
        self._reward_history: list[tuple[RewardComponents, float]] = []
        self._reward_prior = np.array([0.45, 0.25, 0.15, 0.15], dtype=np.float64)
        self._intent_centroids: dict[int, NDArray[np.float64]] = {}
        self._ece_before: Optional[float] = None
        self._ece_after: Optional[float] = None
        self._epoch_accuracy: list[float] = []
        self._conditional_accuracy: Optional[NDArray[np.float64]] = None
        self._feature_indices: dict[str, int] = {}
        self.optimized = bool(self.config.optimized)
        self.cache_size = max(int(self.config.cache_size), 0)
        self.enable_cache = self.optimized and self.cache_size > 0
        self._vector_cache: OrderedDict[str, tuple[np.ndarray, dict[str, float]]] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._feature_keywords: dict[str, tuple[str, ...]] = {
            "__feat_definition_phrase": (
                " what is ",
                " define ",
                " definition of ",
                " explain ",
                " meaning of ",
                " clarify ",
            ),
            "__feat_how_to_phrase": (
                " how do ",
                " how to ",
                " steps to ",
                " procedure for ",
                " method to ",
                " best way to ",
            ),
            "__feat_comparison_phrase": (
                " compare ",
                " difference between ",
                " versus ",
                " vs ",
                " contrast ",
                " differ from ",
            ),
            "__feat_exploration_phrase": (
                " explore ",
                " brainstorm ",
                " ideas for ",
                " creative ideas ",
                " creative ways ",
                " inspiration for ",
                " possibilities for ",
            ),
            "__feat_reflective_language": (
                " reflect ",
                " journal ",
                " mindful ",
                " introspect ",
                " consider deeply ",
                " self-awareness ",
            ),
        }
        self._feature_names: tuple[str, ...] = tuple(
            sorted({"__feat_question_mark", *self._feature_keywords.keys()})
        )

    # Training --------------------------------------------------------------------
    def fit(self, examples: Iterable[IntentExample]) -> None:
        dataset = list(examples)
        if not dataset:
            raise ValueError("No examples supplied")
        self._reset_cache()
        self._reward_history = []
        self._weights_for_inference = None
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
        self._epoch_accuracy = []
        base_step = self.config.learning_rate
        step_scale = max(_safe_step_scale(matrix), 1.0)
        if step_scale > _STEP_SCALE_THRESHOLD:
            effective_scale = math.log(step_scale + 1.0)
            if effective_scale < 1.0:
                effective_scale = 1.0
            effective_scale = min(effective_scale, 1.5)
            step = base_step / effective_scale
        else:
            step = base_step
        min_step = step * 1e-3 if step else 0.0
        l2 = max(float(self.config.l2_regularization), 0.0)

        for epoch in range(self.config.epochs):
            logits = np.asarray(matrix @ self.weights, dtype=float)
            probs = self._softmax(logits)
            one_hot = np.eye(num_labels, dtype=float)[labels]
            gradient = matrix.T @ (probs - one_hot) / len(dataset)
            if l2:
                gradient += l2 * self.weights

            update = step * gradient
            new_weights = self.weights - update
            if not np.isfinite(new_weights).all():
                if step <= min_step or step == 0.0:
                    LOGGER.warning(
                        "Intent training halted early due to non-finite weight update"
                    )
                    break
                step *= 0.5
                continue

            self.weights = new_weights
            loss = -np.mean(
                np.log(probs[np.arange(len(dataset)), labels] + 1e-12)
            )
            accuracy = float(np.mean(np.argmax(probs, axis=1) == labels))
            self._epoch_accuracy.append(accuracy)
            LOGGER.debug("Intent epoch %s | loss=%.4f", epoch + 1, loss)
        LOGGER.info("Trained intent classifier with %d intents", num_labels)
        self._post_train_adjustments(matrix, labels, dataset, probs)
        self._finalise_weights()

    def _prepare_labels(self, examples: Sequence[IntentExample]) -> None:
        for example in examples:
            if example.intent not in self.label_to_index:
                index = len(self.label_to_index)
                self.label_to_index[example.intent] = index
                self.index_to_label[index] = example.intent

    def _build_vocabulary(self, texts: Sequence[Sequence[str]]) -> None:
        token_set = {token for tokens in texts for token in tokens}
        vocab = sorted(token_set)
        for feature in self._feature_names:
            if feature not in token_set:
                vocab.append(feature)
        self.vocabulary = {token: idx for idx, token in enumerate(vocab)}
        self._feature_indices = {
            feature: self.vocabulary[feature]
            for feature in self._feature_names
            if feature in self.vocabulary
        }
        LOGGER.debug("Built vocabulary of size %d", len(self.vocabulary))

    def _vectorise(self, examples: Sequence[IntentExample]) -> NDArray[np.float64]:
        tokenised = [tokenize(example.text) for example in examples]
        if not self.vocabulary:
            self._build_vocabulary(tokenised)
        matrix = np.zeros((len(tokenised), len(self.vocabulary)), dtype=float)
        for row, (tokens, example) in enumerate(zip(tokenised, examples)):
            for token in tokens:
                if token in self.vocabulary:
                    matrix[row, self.vocabulary[token]] += 1.0
            for name, value in self._feature_activations(example.text).items():
                index = self._feature_indices.get(name)
                if index is not None:
                    matrix[row, index] = value
        return _normalise_design_matrix(matrix)

    # Prediction ------------------------------------------------------------------
    def predict(self, text: str) -> str:
        probabilities = self.predict_proba(text)
        return max(probabilities, key=lambda label: probabilities[label])

    def predict_proba(self, text: str) -> dict[str, float]:
        weights = self._weights_for_inference
        if weights is None:
            raise ValueError("Classifier has not been trained")
        weights64 = np.asarray(weights, dtype=np.float64)
        features = self._feature_activations(text)
        fast: Optional[NDArray[np.float64]] = (
            self._fast_path_distribution(features) if self.optimized else None
        )
        vector: Optional[np.ndarray] = None
        used_fast = fast is not None
        if used_fast:
            assert fast is not None  # narrow type for mypy
            probs: NDArray[np.float64] = np.asarray(fast, dtype=np.float64)
        else:
            vector, computed_features = self._vectorise_with_features(text)
            features = computed_features
            if self.optimized:
                indices = np.nonzero(vector)[0]
                if indices.size == 0:
                    logits = np.zeros(len(self.index_to_label), dtype=np.float64)
                else:
                    values = np.asarray(vector[indices], dtype=np.float64)
                    weights_slice = np.asarray(weights64[indices], dtype=np.float64)
                    logits = weights_slice.T @ values
                logits64 = np.asarray(logits, dtype=np.float64)
                probs = self._softmax(logits64[np.newaxis, :])[0]
            else:
                logits = np.asarray(vector @ weights64, dtype=np.float64)
                probs = self._softmax(logits[np.newaxis, :])[0]
        if not used_fast:
            probs = self._apply_systematic_correction(probs)
            probs = self._apply_dirichlet_calibration(probs)
        probs = project_to_simplex(np.asarray(probs, dtype=np.float64))
        if vector is None:
            similarities = np.zeros(len(self.index_to_label), dtype=float)
        else:
            similarities = np.array(
                [
                    self._semantic_similarity(text, index)
                    for index in range(len(self.index_to_label))
                ]
            )
        if vector is not None and np.any(similarities > 0):
            weighting = np.exp(2.0 * (similarities - 0.5))
            probs = project_to_simplex(probs * weighting)
        bias = self._intent_bias(features)
        if bias is not None:
            probs = project_to_simplex(probs * bias)
        return {self.index_to_label[i]: float(prob) for i, prob in enumerate(probs)}

    def reward(
        self,
        text: str,
        selected_intent: str,
        optimal_intent: str,
        feedback: float,
    ) -> float:
        """Return the composite reward for ``selected_intent`` on ``text``."""

        components = self.reward_components(text, selected_intent, optimal_intent, feedback)
        return composite_reward(components, self.reward_weights.tolist())

    def reward_components(
        self,
        text: str,
        selected_intent: str,
        optimal_intent: str,
        feedback: float,
    ) -> RewardComponents:
        """Compute the reward component vector for the selected action."""

        if selected_intent not in self.label_to_index:
            raise ValueError(f"Unknown intent: {selected_intent}")
        if optimal_intent not in self.label_to_index:
            raise ValueError(f"Unknown optimal intent: {optimal_intent}")
        probabilities = self.predict_proba(text)
        prob_selected = probabilities[selected_intent]
        selected_idx = self.label_to_index[selected_intent]
        optimal_idx = self.label_to_index[optimal_intent]
        semantic = self._semantic_similarity(text, selected_idx)
        return RewardComponents(
            correctness=correctness_reward(selected_idx, optimal_idx),
            confidence=confidence_reward(prob_selected, selected_idx, optimal_idx),
            semantic=semantic_similarity_reward(semantic),
            feedback=feedback_reward(feedback),
        )

    def register_feedback(
        self,
        text: str,
        selected_intent: str,
        optimal_intent: str,
        feedback: float,
    ) -> NDArray[np.float64]:
        """Update the reward weights with a new feedback observation."""

        components = self.reward_components(text, selected_intent, optimal_intent, feedback)
        self._reward_history.append((components, feedback))
        self._recompute_reward_weights()
        return self.reward_weights

    @property
    def reward_weights(self) -> NDArray[np.float64]:
        """Return the learned composite reward weights."""

        if self._reward_weights is None:
            return np.full(4, 0.25, dtype=np.float64)
        return np.asarray(self._reward_weights, dtype=np.float64).copy()

    def set_cache_enabled(self, enabled: bool) -> None:
        """Toggle the inference cache used during vectorisation."""

        self.enable_cache = bool(enabled) and self.cache_size > 0
        if not self.enable_cache:
            self._reset_cache()

    def cache_metrics(self) -> tuple[int, int]:
        """Return cache hit and miss counts."""

        return self._cache_hits, self._cache_misses

    @property
    def calibration_report(self) -> tuple[float, float]:
        """Return the pre- and post-calibration expected calibration error."""

        if self._ece_before is None or self._ece_after is None:
            raise ValueError("Calibration report is unavailable before training")
        return self._ece_before, self._ece_after

    @property
    def training_accuracy_curve(self) -> list[float]:
        """Return the recorded accuracy trajectory over training epochs."""

        return list(self._epoch_accuracy)

    # Internal helpers -----------------------------------------------------------
    def _post_train_adjustments(
        self,
        matrix: NDArray[np.float64],
        labels: NDArray[np.int_],
        dataset: Sequence[IntentExample],
        probs: NDArray[np.float64],
    ) -> None:
        num_labels = len(self.label_to_index)
        self._calibrator = DirichletCalibrator(alpha=[1.0] * num_labels)
        for label in labels:
            self._calibrator.update(int(label))
        posterior = self._calibrator.posterior_predictive().predictive
        self._posterior_predictive = np.asarray(posterior, dtype=np.float64)

        confusion = np.zeros((num_labels, num_labels), dtype=np.float64)
        predicted_indices = np.argmax(probs, axis=1)
        for true_idx, pred_idx in zip(labels, predicted_indices):
            confusion[true_idx, pred_idx] += 1.0

        svd_correction = compute_confusion_correction(confusion)
        conditional = np.zeros_like(confusion)
        column_sums = confusion.sum(axis=0, keepdims=True)
        np.divide(confusion, column_sums, out=conditional, where=column_sums > 0)
        combined = 0.5 * svd_correction + 0.5 * conditional
        for col in range(combined.shape[1]):
            column = np.clip(combined[:, col], 0.0, None)
            if column.sum() == 0.0 and column_sums[0, col] > 0.0:
                column = conditional[:, col]
            combined[:, col] = project_to_simplex(column)
        self._correction_matrix = np.asarray(combined, dtype=np.float64)
        self._conditional_accuracy = np.asarray(
            np.clip(np.diag(conditional), 0.0, 1.0),
            dtype=np.float64,
        )

        corrected_probs = np.clip((self._correction_matrix @ probs.T).T, 0.0, None)
        calibrated = np.array([self._posterior_mix(row) for row in corrected_probs])
        self._ece_before = self._expected_calibration_error(probs, labels)
        self._ece_after = self._expected_calibration_error(calibrated, labels)

        self._intent_centroids = self._compute_intent_centroids(matrix, labels)
        history: list[RewardComponents] = []
        realised: list[float] = []
        for row, example in enumerate(dataset):
            selected = int(np.argmax(calibrated[row]))
            prob_selected = float(calibrated[row, selected])
            optimal = int(labels[row])
            semantic = self._semantic_similarity_from_vector(matrix[row], selected)
            components = RewardComponents(
                correctness=correctness_reward(selected, optimal),
                confidence=confidence_reward(prob_selected, selected, optimal),
                semantic=semantic_similarity_reward(semantic),
                feedback=feedback_reward(example.feedback),
            )
            history.append(components)
            realised.append(example.feedback)

        self._reward_history = list(zip(history, realised))
        self._recompute_reward_weights()

    def _apply_systematic_correction(self, probs: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._correction_matrix is None:
            return probs
        correction = np.asarray(self._correction_matrix, dtype=np.float64)
        corrected = correction @ probs
        return np.clip(corrected, 0.0, None)

    def _apply_dirichlet_calibration(self, probs: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._calibrator is None:
            return probs
        calibrated = self._posterior_mix(probs)
        return np.clip(calibrated, 0.0, None)

    def _posterior_mix(self, probs: NDArray[np.float64]) -> NDArray[np.float64]:
        posterior = self._posterior_predictive
        adjusted = np.asarray(probs, dtype=np.float64)
        if self._conditional_accuracy is not None:
            adjusted = project_to_simplex(
                np.maximum(adjusted, np.asarray(self._conditional_accuracy, dtype=np.float64))
            )
        if posterior is None and self._calibrator is not None:
            posterior = self._calibrator.posterior_predictive().predictive
        if posterior is None:
            return adjusted
        posterior_array = np.asarray(posterior, dtype=np.float64)
        weighted = adjusted * posterior_array
        weighted = np.clip(weighted, 0.0, None)
        total = weighted.sum()
        if total == 0.0:
            weighted = posterior_array
        else:
            weighted /= total
        return np.asarray(weighted, dtype=np.float64)

    def _vectorise_text(self, text: str) -> np.ndarray:
        vector, _ = self._vectorise_with_features(text)
        return vector

    def _vectorise_with_features(self, text: str) -> tuple[np.ndarray, dict[str, float]]:
        if not self.vocabulary:
            raise ValueError("Classifier has not been trained")
        cache_key = normalise_text(text)
        if self.enable_cache:
            cached = self._vector_cache.get(cache_key)
            if cached is not None:
                self._cache_hits += 1
                self._vector_cache.move_to_end(cache_key)
                cached_vector, cached_features = cached
                return cached_vector.copy(), dict(cached_features)
        dtype = np.float16 if self.optimized else np.float64
        working = np.zeros((len(self.vocabulary),), dtype=np.float64)
        for token in tokenize(text):
            index = self.vocabulary.get(token)
            if index is not None:
                working[index] += 1.0
        features = self._feature_activations(text)
        for name, value in features.items():
            index = self._feature_indices.get(name)
            if index is not None:
                working[index] = value
        vector64 = _normalise_feature_vector(working)
        if dtype is np.float64:
            vector = vector64
        else:
            vector = np.asarray(vector64, dtype=dtype)
        self._cache_misses += 1
        if self.enable_cache:
            self._cache_insert(cache_key, vector, features)
        return vector.copy(), dict(features)

    def _feature_activations(self, text: str) -> dict[str, float]:
        normalised = f" {normalise_text(text)} "
        features: dict[str, float] = {}
        for name, keywords in self._feature_keywords.items():
            if any(keyword in normalised for keyword in keywords):
                features[name] = 1.0
        if "?" in text:
            features["__feat_question_mark"] = 1.0
        return features

    def _intent_bias(self, features: dict[str, float]) -> Optional[NDArray[np.float64]]:
        if not features:
            return None
        num_labels = len(self.index_to_label)
        bias = np.ones(num_labels, dtype=np.float64)
        changed = False

        def bump(intent: str, factor: float) -> None:
            nonlocal changed
            index = self.label_to_index.get(intent)
            if index is None:
                return
            bias[index] *= factor
            changed = True

        has_definition_phrase = bool(features.get("__feat_definition_phrase"))
        has_how_to_phrase = bool(features.get("__feat_how_to_phrase"))
        if has_definition_phrase:
            bump("definition", 1.35)
            bump("how_to", 0.8)
        if has_how_to_phrase:
            bump("how_to", 1.35)
            bump("definition", 0.8)
        if has_definition_phrase and has_how_to_phrase:
            bump("how_to", 1.15)
            bump("definition", 0.75)
        if features.get("__feat_comparison_phrase"):
            bump("comparison", 1.35)
            bump("definition", 0.85)
        if features.get("__feat_exploration_phrase") or features.get("__feat_reflective_language"):
            bump("exploration", 1.35)
        if features.get("__feat_question_mark") and not features.get("__feat_how_to_phrase"):
            bump("definition", 1.1)
        return np.asarray(bias, dtype=np.float64) if changed else None

    def _fast_path_distribution(self, features: dict[str, float]) -> Optional[NDArray[np.float64]]:
        if not features:
            return None
        mapping = {
            "__feat_definition_phrase": "definition",
            "__feat_how_to_phrase": "how_to",
            "__feat_comparison_phrase": "comparison",
            "__feat_exploration_phrase": "exploration",
        }
        triggered = [intent for feature, intent in mapping.items() if features.get(feature)]
        if len(triggered) != 1:
            return None
        intent = triggered[0]
        index = self.label_to_index.get(intent)
        if index is None:
            return None
        probs = np.full(len(self.index_to_label), 1e-4, dtype=np.float64)
        probs[index] = 0.9996
        return project_to_simplex(probs)

    def _cache_insert(self, key: str, vector: np.ndarray, features: dict[str, float]) -> None:
        if not self.enable_cache or self.cache_size == 0:
            return
        if len(self._vector_cache) >= self.cache_size:
            self._vector_cache.popitem(last=False)
        self._vector_cache[key] = (vector.copy(), dict(features))
        self._vector_cache.move_to_end(key)

    def _reset_cache(self) -> None:
        self._vector_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def _finalise_weights(self) -> None:
        if self.weights is None:
            return
        if self.optimized:
            optimised_weights = np.asarray(self.weights, dtype=np.float16)
            self.weights = optimised_weights
            self._weights_for_inference = np.asarray(optimised_weights, dtype=np.float64)
        else:
            precise = np.asarray(self.weights, dtype=np.float64)
            self.weights = precise
            self._weights_for_inference = precise.copy()

    def _blend_reward_weights(self, learned: NDArray[np.float64]) -> NDArray[np.float64]:
        prior_weight = float(np.clip(self.config.feedback_prior_weight, 0.0, 1.0))
        candidate = project_to_simplex(
            (1.0 - prior_weight) * learned + prior_weight * self._reward_prior
        )
        if self._reward_weights is None:
            return np.asarray(candidate, dtype=np.float64)
        step = float(np.clip(self.config.feedback_step_size, 0.0, 1.0))
        updated = (1.0 - step) * self._reward_weights + step * candidate
        return project_to_simplex(np.asarray(updated, dtype=np.float64))

    def _recompute_reward_weights(self) -> None:
        if not self._reward_history:
            return
        components, rewards = zip(*self._reward_history)
        learned = np.asarray(
            estimate_optimal_weights(list(components), list(rewards)),
            dtype=np.float64,
        )
        self._reward_weights = self._blend_reward_weights(learned)

    def _compute_intent_centroids(
        self, matrix: NDArray[np.float64], labels: NDArray[np.int_]
    ) -> dict[int, NDArray[np.float64]]:
        centroids: dict[int, NDArray[np.float64]] = {}
        for index in range(len(self.label_to_index)):
            mask = labels == index
            if not np.any(mask):
                centroids[index] = np.zeros(matrix.shape[1], dtype=np.float64)
                continue
            centroid = np.asarray(matrix[mask].mean(axis=0), dtype=np.float64)
            centroids[index] = centroid
        return centroids

    def _semantic_similarity(self, text: str, intent_index: int) -> float:
        vector = self._vectorise_text(text)
        return self._semantic_similarity_from_vector(vector, intent_index)

    def _semantic_similarity_from_vector(
        self, vector: NDArray[np.float64], intent_index: int
    ) -> float:
        centroid = self._intent_centroids.get(intent_index)
        if centroid is None or np.allclose(centroid, 0.0):
            return 0.0
        numerator = float(vector @ centroid)
        denominator = float(np.linalg.norm(vector) * np.linalg.norm(centroid))
        if denominator == 0.0:
            return 0.0
        cosine = numerator / denominator
        return 0.5 * (cosine + 1.0)

    @staticmethod
    def _expected_calibration_error(
        probs: NDArray[np.float64], labels: NDArray[np.int_], num_bins: int = 10
    ) -> float:
        confidences = probs.max(axis=1)
        predictions = np.argmax(probs, axis=1)
        bins = np.linspace(0.0, 1.0, num_bins + 1)
        ece = 0.0
        total = len(confidences)
        for lower, upper in zip(bins[:-1], bins[1:]):
            mask = (confidences >= lower) & (confidences < upper)
            if not np.any(mask):
                continue
            bin_confidence = confidences[mask].mean()
            bin_accuracy = np.mean(predictions[mask] == labels[mask])
            ece += abs(bin_accuracy - bin_confidence) * np.sum(mask) / total
        return float(ece)

    @staticmethod
    def _softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        denominator = np.sum(exp, axis=1, keepdims=True)
        return np.asarray(exp / denominator, dtype=float)

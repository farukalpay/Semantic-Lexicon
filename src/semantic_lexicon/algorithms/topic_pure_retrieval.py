# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""Topic-pure top-k retrieval with a learned metric and gating vector.

This module implements the optimisation problem described in the prompt: we learn a
positive semi-definite metric ``M`` together with a dimension-wise gate ``g`` that focuses
query representations onto topic-relevant subspaces. The training objective enforces
margin-separated triplets (query, positive concept, negative concept) while regularising
``M`` towards the identity and promoting sparsity in ``g``.

The workflow is:

1. **Isotropy fix.** Apply whitening to concept and query embeddings followed by
   unit-norm normalisation to remove dominant directions and norm bias.
2. **Representation.** Combine the whitened query with an optional persona vector and
   gate it element-wise by ``g``.
3. **Metric learning.** Optimise ``M`` (PSD) and ``g`` with a triplet hinge loss and
   the regularisers from the specification.
4. **Inference.** Retrieve the top-k concepts by the learned similarity ``s(q, c)``.

The implementation favours clarity over raw performance—the dataset sizes used in the
tests are modest, so a straightforward numpy-based optimiser is sufficient.

Helper routines `_normalise_rows`, `_compute_covariance`, and `_project_to_psd` accept
``float64`` inputs and always return ``float64`` arrays. Label encoders convert any
string labels into contiguous ``int64`` identifiers so downstream metrics and
constraints operate on a consistent numeric domain.
"""

from __future__ import annotations

from collections import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from ._typing import FloatArray, IntArray
else:  # pragma: no cover - runtime aliases for type checkers
    FloatArray = Any
    IntArray = Any

EPSILON = 1e-8


def _normalise_rows(matrix: FloatArray) -> FloatArray:
    """Return a row-normalised float64 matrix.

    Parameters
    ----------
    matrix:
        A two-dimensional ``float64`` array whose rows will be scaled to unit
        length.

    Returns
    -------
    FloatArray
        The normalised matrix with ``float64`` dtype preserved.
    """

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, EPSILON)
    normalised = (matrix / norms).astype(np.float64, copy=False)
    return cast("FloatArray", normalised)


def _ensure_rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(seed)


def _symmetric_outer(u: FloatArray, v: FloatArray) -> FloatArray:
    """Compute a symmetric outer product with float64 precision."""

    symmetric = 0.5 * (np.outer(u, v) + np.outer(v, u))
    return cast("FloatArray", symmetric.astype(np.float64, copy=False))


def _project_to_psd(matrix: FloatArray) -> FloatArray:
    """Project ``matrix`` onto the PSD cone while preserving ``float64`` dtype."""

    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, 0.0, None)
    projected = (eigvecs * eigvals) @ eigvecs.T
    projected = 0.5 * (projected + projected.T)
    return cast("FloatArray", projected.astype(np.float64, copy=False))


def _compute_covariance(matrix: FloatArray) -> FloatArray:
    """Compute the float64 covariance matrix of ``matrix`` rows."""

    if matrix.shape[0] <= 1:
        return cast("FloatArray", np.eye(matrix.shape[1], dtype=np.float64))
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    cov = centered.T @ centered / float(matrix.shape[0])
    return cast("FloatArray", cov.astype(np.float64, copy=False))


def _standardise_labels(
    labels: np.ndarray[Any, np.dtype[np.generic]],
) -> tuple[IntArray, dict[object, int]]:
    """Encode arbitrary labels as contiguous ``int64`` identifiers."""

    flat = np.asarray(labels)
    if flat.ndim != 1:
        raise ValueError("Labels must be a one-dimensional array")
    ordered_unique = list(dict.fromkeys(flat.tolist()))
    mapping = {label: idx for idx, label in enumerate(ordered_unique)}
    encoded = np.array([mapping[label] for label in flat.tolist()], dtype=np.int64)
    return cast("IntArray", encoded), mapping


def _encode_with_mapping(
    labels: np.ndarray[Any, np.dtype[np.generic]], mapping: abc.Mapping[object, int]
) -> IntArray:
    """Map ``labels`` to ``int64`` ids using ``mapping``."""

    flat = np.asarray(labels)
    if flat.ndim != 1:
        raise ValueError("Labels must be a one-dimensional array")
    try:
        encoded = np.array([mapping[label] for label in flat.tolist()], dtype=np.int64)
    except KeyError as error:  # pragma: no cover - defensive for unexpected inputs
        raise ValueError(f"Unknown label encountered: {error.args[0]!r}") from error
    return cast("IntArray", encoded)


def _kmeans(
    matrix: FloatArray,
    num_clusters: int,
    rng: np.random.Generator,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> tuple[IntArray, FloatArray]:
    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive")
    num_clusters = min(num_clusters, matrix.shape[0])
    if num_clusters == 0:
        raise ValueError("matrix must contain at least one vector")
    indices = rng.choice(matrix.shape[0], size=num_clusters, replace=False)
    centroids = matrix[indices].astype(np.float64, copy=True)
    labels = np.zeros(matrix.shape[0], dtype=np.int64)
    for _ in range(max_iter):
        diff = matrix[:, None, :] - centroids[None, :, :]
        distances = np.linalg.norm(diff, axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for k in range(num_clusters):
            members = matrix[labels == k]
            if members.size:
                centroids[k] = np.mean(members, axis=0)
    # final update for convergence tolerance
    prev_centroids = centroids.copy()
    for k in range(num_clusters):
        members = matrix[labels == k]
        if members.size:
            centroids[k] = np.mean(members, axis=0)
    shift = np.max(np.linalg.norm(centroids - prev_centroids, axis=1)) if num_clusters else 0.0
    if shift > tol:
        # run one more refinement step when centroids moved substantially
        for k in range(num_clusters):
            members = matrix[labels == k]
            if members.size:
                centroids[k] = np.mean(members, axis=0)
    return cast("IntArray", labels.astype(np.int64, copy=False)), cast(
        "FloatArray", centroids.astype(np.float64, copy=False)
    )


@dataclass
class TopicPureRetrievalConfig:
    """Configuration for the topic-pure retrieval trainer."""

    k: int = 5
    margin: float = 0.3
    lambda_reg: float = 1e-3
    beta_reg: float = 1e-3
    learning_rate: float = 0.05
    epochs: int = 100
    negative_samples: int = 5
    cluster_count: int | None = None
    random_state: int | np.random.Generator | None = None


@dataclass
class TrainingStats:
    epoch: int
    triplet_loss: float
    regularisation_loss: float
    total_loss: float
    violation_rate: float


class TopicPureRetriever:
    """Learn a PSD metric and gating vector for topic-pure retrieval."""

    def __init__(self, config: TopicPureRetrievalConfig):
        self.config = config
        self.rng = _ensure_rng(config.random_state)
        self.concept_ids_: tuple[str, ...] | None = None
        self.query_ids_: tuple[str, ...] | None = None
        self.mean_: FloatArray | None = None
        self.whitener_: FloatArray | None = None
        self.persona_vector_: FloatArray | None = None
        self.concept_embeddings_: FloatArray | None = None
        self.query_embeddings_: FloatArray | None = None
        self.query_representations_: FloatArray | None = None
        self.M_: FloatArray | None = None
        self.gate_: FloatArray | None = None
        self.history_: list[TrainingStats] = []
        self.concept_labels_: IntArray | None = None
        self.query_labels_: IntArray | None = None
        self.label_mapping_: dict[object, int] | None = None
        self._concept_index: dict[str, int] = {}
        self._query_index: dict[str, int] = {}
        self.pre_whiten_condition_number_: float | None = None
        self.post_whiten_condition_number_: float | None = None
        self.whitened_concepts_: FloatArray | None = None
        self.whitened_queries_: FloatArray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self,
        concept_ids: abc.Sequence[str],
        concept_embeddings: FloatArray,
        query_ids: abc.Sequence[str],
        query_embeddings: FloatArray,
        concept_labels: abc.Sequence[str] | None = None,
        query_labels: abc.Sequence[str] | None = None,
        persona: FloatArray | None = None,
    ) -> TopicPureRetriever:
        """Fit the metric and gate using the provided embeddings."""

        concept_embeddings = cast("FloatArray", np.asarray(concept_embeddings, dtype=np.float64))
        query_embeddings = cast("FloatArray", np.asarray(query_embeddings, dtype=np.float64))
        if not isinstance(concept_ids, abc.Sequence) or not isinstance(query_ids, abc.Sequence):
            raise TypeError("concept_ids and query_ids must be sequences")
        if concept_embeddings.ndim != 2 or query_embeddings.ndim != 2:
            raise ValueError("Embeddings must be two-dimensional arrays")
        if concept_embeddings.shape[1] != query_embeddings.shape[1]:
            raise ValueError("Concepts and queries must share the same embedding dimension")
        if concept_embeddings.shape[0] != len(concept_ids):
            raise ValueError("Number of concept ids does not match embeddings")
        if query_embeddings.shape[0] != len(query_ids):
            raise ValueError("Number of query ids does not match embeddings")

        self.concept_ids_ = tuple(concept_ids)
        self.query_ids_ = tuple(query_ids)
        self._concept_index = {cid: i for i, cid in enumerate(self.concept_ids_)}
        self._query_index = {qid: i for i, qid in enumerate(self.query_ids_)}

        persona_array: FloatArray | None
        if persona is None:
            persona_array = None
        else:
            persona_array = cast("FloatArray", np.asarray(persona, dtype=np.float64))

        self._preprocess_embeddings(concept_embeddings, query_embeddings, persona_array)

        if concept_labels is None:
            concept_labels_array, centroids = self._cluster_concepts()
            concept_labels_array = cast(
                "IntArray", concept_labels_array.astype(np.int64, copy=False)
            )
            unique_labels = [int(label) for label in np.unique(concept_labels_array)]
            self.label_mapping_ = {label: label for label in unique_labels}
        else:
            if not isinstance(concept_labels, (abc.Sequence, np.ndarray)):
                raise TypeError("concept_labels must be a sequence when provided")
            concept_labels_array, label_mapping = _standardise_labels(np.asarray(concept_labels))
            self.label_mapping_ = label_mapping
            centroids = self._cluster_centroids_from_labels(concept_labels_array)
        if concept_labels_array.shape[0] != len(self.concept_ids_):
            raise ValueError("Concept labels must match the number of concepts")
        self.concept_labels_ = concept_labels_array

        if query_labels is None:
            self.query_labels_ = self._assign_query_labels(centroids)
        else:
            if not isinstance(query_labels, (abc.Sequence, np.ndarray)):
                raise TypeError("query_labels must be a sequence when provided")
            query_labels_array = np.asarray(query_labels)
            if query_labels_array.shape[0] != len(self.query_ids_):
                raise ValueError("Query labels must match the number of queries")
            if self.label_mapping_ is None:
                raise RuntimeError("Label mapping missing")
            self.query_labels_ = _encode_with_mapping(query_labels_array, self.label_mapping_)

        self._optimise()
        self._refresh_query_representations()
        return self

    def top_k_for_query_id(
        self,
        query_id: str,
        k: int | None = None,
    ) -> list[tuple[str, float]]:
        self._validate_fitted()
        if self.query_representations_ is None:
            raise RuntimeError("Query representations missing")
        if self.concept_ids_ is None:
            raise RuntimeError("Concept identifiers missing")
        if query_id not in self._query_index:
            raise KeyError(f"Unknown query id: {query_id}")
        idx = self._query_index[query_id]
        representation = self.query_representations_[idx]
        scores = self._score_representation(representation)
        order = np.argsort(scores)[::-1]
        k = self._resolve_k(k)
        top = order[:k]
        return [(self.concept_ids_[i], float(scores[i])) for i in top]

    def top_k(
        self,
        query_embedding: FloatArray,
        persona: FloatArray | None = None,
        k: int | None = None,
    ) -> list[tuple[str, float]]:
        """Retrieve top-k concepts for an arbitrary query embedding."""

        self._validate_fitted()
        if self.concept_ids_ is None:
            raise RuntimeError("Concept identifiers missing")
        query_embedding = np.asarray(query_embedding, dtype=np.float64)
        if query_embedding.ndim != 1:
            raise ValueError("query_embedding must be a one-dimensional vector")
        z = self._apply_whitening(query_embedding[None, :])[0]
        if persona is None:
            if self.persona_vector_ is None:
                raise RuntimeError("Persona vector missing")
            persona_vector = self.persona_vector_
        else:
            persona_vector = self._whiten_persona(persona)
        representation = self._build_representation(z, persona_vector)
        scores = self._score_representation(representation)
        order = np.argsort(scores)[::-1]
        k = self._resolve_k(k)
        top = order[:k]
        return [(self.concept_ids_[i], float(scores[i])) for i in top]

    def purity_at_k(self, query_id: str, k: int | None = None) -> float:
        """Compute the topic purity diagnostic for ``query_id``."""

        self._validate_fitted()
        if self.concept_ids_ is None:
            raise RuntimeError("Concept identifiers missing")
        if self.query_labels_ is None or self.concept_labels_ is None:
            raise ValueError("Purity requires labels or clustered assignments")
        top = self.top_k_for_query_id(query_id, k)
        q_label = self.query_labels_[self._query_index[query_id]]
        matches = 0
        for concept_id, _ in top:
            idx = self._concept_index[concept_id]
            if self.concept_labels_[idx] == q_label:
                matches += 1
        return matches / float(len(top)) if top else 0.0

    @property
    def gate_sparsity(self) -> float:
        self._validate_fitted()
        if self.gate_ is None:
            raise RuntimeError("Gate vector missing")
        non_zero = np.count_nonzero(self.gate_ > 1e-6)
        return non_zero / float(self.gate_.size)

    @property
    def triplet_violation_rate(self) -> float:
        if not self.history_:
            return 0.0
        return self.history_[-1].violation_rate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_fitted(self) -> None:
        if (
            self.concept_ids_ is None
            or self.query_ids_ is None
            or self.concept_embeddings_ is None
            or self.query_representations_ is None
            or self.M_ is None
            or self.gate_ is None
        ):
            raise RuntimeError("TopicPureRetriever must be fitted before use")

    def _resolve_k(self, k: int | None) -> int:
        if self.concept_ids_ is None:
            raise RuntimeError("Concept identifiers missing")
        size = len(self.concept_ids_)
        target = self.config.k if k is None else k
        if target <= 0:
            raise ValueError("k must be positive")
        return min(target, size)

    def _preprocess_embeddings(
        self,
        concept_embeddings: FloatArray,
        query_embeddings: FloatArray,
        persona: FloatArray | None,
    ) -> None:
        self.mean_ = np.mean(concept_embeddings, axis=0)
        cov = _compute_covariance(concept_embeddings)
        cov += EPSILON * np.eye(cov.shape[0])
        self.pre_whiten_condition_number_ = float(np.linalg.cond(cov))
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, EPSILON, None)
        inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        self.whitener_ = inv_sqrt

        whitened_concepts = self._apply_whitening(concept_embeddings)
        whitened_concepts -= np.mean(whitened_concepts, axis=0, keepdims=True)
        whitened_queries = self._apply_whitening(query_embeddings)
        whitened_queries -= np.mean(whitened_queries, axis=0, keepdims=True)
        self.whitened_concepts_ = whitened_concepts
        self.whitened_queries_ = whitened_queries

        self.concept_embeddings_ = _normalise_rows(whitened_concepts)
        self.query_embeddings_ = _normalise_rows(whitened_queries)

        whitened_unscaled = (concept_embeddings - self.mean_) @ self.whitener_.T
        cov_after = _compute_covariance(whitened_unscaled)
        cov_after += EPSILON * np.eye(cov_after.shape[0])
        self.post_whiten_condition_number_ = float(np.linalg.cond(cov_after))

        if persona is None:
            self.persona_vector_ = cast(
                "FloatArray", np.zeros(concept_embeddings.shape[1], dtype=np.float64)
            )
        else:
            self.persona_vector_ = self._whiten_persona(persona)

    def _apply_whitening(self, matrix: FloatArray) -> FloatArray:
        if self.mean_ is None or self.whitener_ is None:
            raise RuntimeError("Whitening parameters missing")
        centered = matrix - self.mean_
        whitened = (centered @ self.whitener_.T).astype(np.float64, copy=False)
        return cast("FloatArray", whitened)

    def _whiten_persona(self, persona: FloatArray) -> FloatArray:
        persona = cast("FloatArray", np.asarray(persona, dtype=np.float64))
        if persona.ndim != 1:
            raise ValueError("persona vector must be one-dimensional")
        if self.mean_ is None or self.whitener_ is None:
            raise RuntimeError("Whitening parameters missing")
        whitened = ((persona - self.mean_) @ self.whitener_.T).astype(np.float64, copy=False)
        return cast("FloatArray", whitened)

    def _cluster_concepts(self) -> tuple[IntArray, FloatArray]:
        if self.concept_embeddings_ is None:
            raise RuntimeError("Embeddings must be preprocessed before clustering")
        num_clusters = self.config.cluster_count
        if num_clusters is None:
            num_clusters = max(1, int(np.sqrt(self.concept_embeddings_.shape[0])))
        labels, centroids = _kmeans(self.concept_embeddings_, num_clusters, self.rng)
        return labels, centroids

    def _cluster_centroids_from_labels(self, labels: IntArray) -> FloatArray:
        if self.concept_embeddings_ is None:
            raise RuntimeError("Embeddings must be preprocessed before clustering")
        if labels.size == 0:
            raise ValueError("Cannot build centroids with no labels")
        num_clusters = int(labels.max()) + 1
        centroids = np.zeros((num_clusters, self.concept_embeddings_.shape[1]), dtype=np.float64)
        for cluster in range(num_clusters):
            members = self.concept_embeddings_[labels == cluster]
            if members.size == 0:
                raise ValueError(f"Label {cluster} is missing concept assignments")
            centroids[cluster] = np.mean(members, axis=0)
        return cast("FloatArray", centroids)

    def _assign_query_labels(self, centroids: FloatArray) -> IntArray:
        if self.query_embeddings_ is None:
            raise RuntimeError("Embeddings must be preprocessed before assigning labels")
        diff = self.query_embeddings_[:, None, :] - centroids[None, :, :]
        distances = np.linalg.norm(diff, axis=2)
        assignments = np.argmin(distances, axis=1).astype(np.int64)
        return cast("IntArray", assignments)

    def _build_triplets(self) -> list[tuple[int, int, int]]:
        if self.query_labels_ is None or self.concept_labels_ is None:
            raise RuntimeError("Labels must be available for triplet construction")
        triplets: list[tuple[int, int, int]] = []
        for q_idx, label in enumerate(self.query_labels_):
            positive_indices = np.where(self.concept_labels_ == label)[0]
            negative_indices = np.where(self.concept_labels_ != label)[0]
            if positive_indices.size == 0 or negative_indices.size == 0:
                continue
            neg_sample_size = min(self.config.negative_samples, negative_indices.size)
            for c_pos in positive_indices:
                neg_samples = self.rng.choice(negative_indices, size=neg_sample_size, replace=False)
                for c_neg in neg_samples:
                    triplets.append((q_idx, c_pos, c_neg))
        return triplets

    def _optimise(self) -> None:
        if self.concept_embeddings_ is None or self.query_embeddings_ is None:
            raise RuntimeError("Embeddings must be preprocessed before optimisation")
        if self.persona_vector_ is None:
            raise RuntimeError("Persona vector missing")
        d = self.concept_embeddings_.shape[1]
        self.M_ = np.eye(d)
        self.gate_ = np.ones(d)
        identity = np.eye(d)
        triplets = self._build_triplets()
        if not triplets:
            self.history_.append(TrainingStats(0, 0.0, 0.0, 0.0, 0.0))
            return

        for epoch in range(1, self.config.epochs + 1):
            grad_M = np.zeros((d, d))
            grad_g = np.zeros(d)
            triplet_loss = 0.0
            violations = 0
            for q_idx, c_pos, c_neg in triplets:
                z = self.query_embeddings_[q_idx]
                z_plus = z + self.persona_vector_
                r = self.gate_ * z_plus
                e_pos = self.concept_embeddings_[c_pos]
                e_neg = self.concept_embeddings_[c_neg]
                s_pos = float(r @ self.M_ @ e_pos)
                s_neg = float(r @ self.M_ @ e_neg)
                margin = self.config.margin - s_pos + s_neg
                if margin > 0:
                    violations += 1
                    triplet_loss += margin
                    grad_M += _symmetric_outer(r, e_neg) - _symmetric_outer(r, e_pos)
                    Me_pos = self.M_ @ e_pos
                    Me_neg = self.M_ @ e_neg
                    grad_g += z_plus * (Me_neg - Me_pos)

            reg_loss = self.config.lambda_reg * float(
                np.linalg.norm(self.M_ - identity) ** 2
            ) + self.config.beta_reg * float(np.sum(np.abs(self.gate_)))
            violation_rate = violations / float(len(triplets)) if triplets else 0.0

            grad_M += 2 * self.config.lambda_reg * (self.M_ - identity)
            grad_g += self.config.beta_reg * np.sign(self.gate_)

            self.M_ -= self.config.learning_rate * grad_M
            self.M_ = 0.5 * (self.M_ + self.M_.T)
            self.M_ = _project_to_psd(self.M_)

            self.gate_ -= self.config.learning_rate * grad_g
            self.gate_ = np.clip(self.gate_, 0.0, 1.0)

            total_loss = triplet_loss + reg_loss
            self.history_.append(
                TrainingStats(
                    epoch=epoch,
                    triplet_loss=triplet_loss,
                    regularisation_loss=reg_loss,
                    total_loss=total_loss,
                    violation_rate=violation_rate,
                )
            )
            if violation_rate < 1e-3:
                break

    def _refresh_query_representations(self) -> None:
        if self.query_embeddings_ is None:
            raise RuntimeError("Query embeddings missing")
        self.query_representations_ = self._build_representations(self.query_embeddings_)

    def _build_representations(self, base_vectors: FloatArray) -> FloatArray:
        if self.persona_vector_ is None or self.gate_ is None:
            raise RuntimeError("Persona or gate parameters missing")
        persona = self.persona_vector_
        representations = (base_vectors + persona) * self.gate_
        return representations

    def _build_representation(self, base_vector: FloatArray, persona: FloatArray) -> FloatArray:
        if self.gate_ is None:
            raise RuntimeError("Gate vector missing")
        return (base_vector + persona) * self.gate_

    def _score_representation(self, representation: FloatArray) -> FloatArray:
        if self.M_ is None or self.concept_embeddings_ is None:
            raise RuntimeError("Scoring parameters missing")
        return representation @ self.M_ @ self.concept_embeddings_.T


__all__ = [
    "TopicPureRetrievalConfig",
    "TopicPureRetriever",
    "TrainingStats",
]

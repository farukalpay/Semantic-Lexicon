# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""Knowledge network management."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Optional, cast

import numpy as np
from numpy.typing import NDArray

from .config import KnowledgeConfig
from .logging import configure_logging
from .utils import tokenize

LOGGER = configure_logging(logger_name=__name__)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class KnowledgeEdge:
    head: str
    relation: str
    tail: str


@dataclass(frozen=True)
class KnowledgeSelection:
    """Summary of a greedy knowledge subset optimisation."""

    concepts: tuple[str, ...]
    relevance: float
    coverage: float
    cohesion: float
    collaboration: float
    diversity: float
    knowledge_raw: float
    gate_mean: float


@dataclass(frozen=True)
class _GreedyCandidate:
    """Payload captured while evaluating a greedy candidate."""

    coverage: FloatArray
    coverage_sum: float
    cohesion_num: float
    cohesion_den: float
    collab_map: dict[int, float]
    delta_collab: float
    feature_vector: FloatArray
    delta_div: float


class KnowledgeNetwork:
    """A light-weight knowledge graph with simple scoring functions."""

    def __init__(self, config: Optional[KnowledgeConfig] = None) -> None:
        self.config = config or KnowledgeConfig()
        self.entities: dict[str, int] = {}
        self.relations: dict[str, int] = {}
        self.embeddings: Optional[FloatArray] = None
        self.relation_matrices: Optional[FloatArray] = None
        self.index_to_entity: list[str] = []
        self.adjacency: Optional[FloatArray] = None
        self.degree: Optional[FloatArray] = None
        self.graph_laplacian: Optional[FloatArray] = None
        self.transition: Optional[FloatArray] = None
        self.similarity: Optional[FloatArray] = None
        self._concept_groups: dict[str, set[str]] = {}
        self._group_bounds: dict[str, tuple[Optional[float], Optional[float]]] = {}
        self._entity_token_cache: dict[str, set[str]] = {}

    # Building --------------------------------------------------------------------
    def _ensure_entity(self, name: str) -> int:
        if name not in self.entities:
            self.entities[name] = len(self.entities)
        return self.entities[name]

    def _ensure_relation(self, name: str) -> int:
        if name not in self.relations:
            self.relations[name] = len(self.relations)
        return self.relations[name]

    def fit(self, edges: Iterable[KnowledgeEdge]) -> None:
        triples = list(edges)
        if not triples:
            raise ValueError("No knowledge edges supplied")
        for triple in triples:
            self._ensure_entity(triple.head)
            self._ensure_entity(triple.tail)
            self._ensure_relation(triple.relation)
        entity_dim = len(self.entities)
        relation_dim = len(self.relations)
        rng = np.random.default_rng(0)
        embeddings = rng.normal(0, 0.1, size=(entity_dim, self.config.max_relations))
        self.embeddings = cast(FloatArray, np.asarray(embeddings, dtype=float))
        relation_tensors = rng.normal(
            0,
            0.1,
            size=(
                relation_dim,
                self.config.max_relations,
                self.config.max_relations,
            ),
        )
        self.relation_matrices = cast(FloatArray, np.asarray(relation_tensors, dtype=float))
        learning_rate = self.config.learning_rate
        for epoch in range(self.config.epochs):
            total_loss = 0.0
            for edge in triples:
                head_idx = self.entities[edge.head]
                tail_idx = self.entities[edge.tail]
                relation_idx = self.relations[edge.relation]
                head_vec = self.embeddings[head_idx]
                tail_vec = self.embeddings[tail_idx]
                relation_matrix = self.relation_matrices[relation_idx]
                score = head_vec @ relation_matrix @ tail_vec
                error = 1.0 - score
                total_loss += error**2
                grad_head = -2 * error * relation_matrix @ tail_vec
                grad_tail = -2 * error * relation_matrix.T @ head_vec
                grad_rel = -2 * error * np.outer(head_vec, tail_vec)
                self.embeddings[head_idx] -= learning_rate * grad_head
                self.embeddings[tail_idx] -= learning_rate * grad_tail
                self.relation_matrices[relation_idx] -= learning_rate * grad_rel
            LOGGER.debug("Knowledge epoch %s | loss=%.4f", epoch + 1, total_loss / len(triples))
        LOGGER.info("Trained knowledge network with %d entities", entity_dim)
        self._build_index_lookup()
        self._build_graph(triples)
        self.similarity = self._compute_similarity_matrix()

    # Metadata --------------------------------------------------------------------
    def set_concept_groups(self, mapping: Mapping[str, Sequence[str]]) -> None:
        """Register concept-to-group assignments used for constrained selection."""

        cleaned: dict[str, set[str]] = {}
        for name, groups in mapping.items():
            group_set = {str(group) for group in groups if str(group)}
            if group_set:
                cleaned[str(name)] = group_set
        self._concept_groups = cleaned

    def set_group_bounds(
        self,
        bounds: Mapping[str, tuple[Optional[float], Optional[float]]],
    ) -> None:
        """Register lower/upper bounds per group.

        Bounds may be absolute counts or fractional ratios interpreted relative to
        the requested selection size during optimisation.
        """

        cleaned: dict[str, tuple[Optional[float], Optional[float]]] = {}
        for name, (lower, upper) in bounds.items():
            cleaned[str(name)] = (
                float(lower) if lower is not None else None,
                float(upper) if upper is not None else None,
            )
        self._group_bounds = cleaned

    # Querying --------------------------------------------------------------------
    def neighbours(self, entity: str, top_k: int = 5) -> list[tuple[str, float]]:
        if entity not in self.entities or not self.index_to_entity:
            return []
        idx = self.entities[entity]
        if self.adjacency is not None and self.adjacency.size:
            scores = self.adjacency[idx]
        elif self.similarity is not None and self.similarity.size:
            scores = self.similarity[idx]
        else:
            return []
        candidates: list[tuple[str, float]] = []
        for other_idx, value in enumerate(scores):
            if other_idx == idx:
                continue
            if value <= 0:
                continue
            name = self.index_to_entity[other_idx]
            candidates.append((name, float(value)))
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:top_k]

    def score(self, head: str, relation: str, tail: str) -> float:
        if (
            self.embeddings is None
            or self.relation_matrices is None
            or head not in self.entities
            or tail not in self.entities
            or relation not in self.relations
        ):
            return 0.0
        h = self.embeddings[self.entities[head]]
        t = self.embeddings[self.entities[tail]]
        r = self.relation_matrices[self.relations[relation]]
        return float(h @ r @ t)

    def select_concepts(
        self,
        prompt_vector: np.ndarray,
        *,
        top_k: Optional[int] = None,
        anchor_tokens: Optional[Sequence[str]] = None,
    ) -> KnowledgeSelection:
        """Select a knowledge subset using the composite objective."""

        if self.embeddings is None or not self.entities:
            return KnowledgeSelection((), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        vector = np.asarray(prompt_vector, dtype=float).ravel()
        if vector.size == 0:
            vector = np.zeros(self.embeddings.shape[1], dtype=float)
        vector = self._align_prompt_vector(vector)
        relevance = self._compute_relevance(vector)
        smoothed = self._smoothed_relevance(relevance)
        topic_weights = self._topic_weights(relevance)
        similarity = self.similarity
        if similarity is None:
            similarity = self._compute_similarity_matrix()
        if similarity is None:
            similarity = np.eye(len(self.entities), dtype=float)
        size = max(1, min(top_k or self.config.selection_size, len(self.entities)))
        anchor_set = self._prepare_anchor_tokens(anchor_tokens or ())
        topic_mask = self._topic_mask(topic_weights)
        anchor_scores: Optional[np.ndarray] = None
        if anchor_set:
            anchor_scores = self._anchor_overlap_scores(anchor_set)
            if anchor_scores.size:
                anchor_hits = anchor_scores > 0.0
                if np.any(anchor_hits):
                    topic_mask = np.logical_or(topic_mask, anchor_hits)
        anchors = self._select_anchors(relevance, topic_mask, size)
        gates, _ = self._compute_anchor_gates(similarity, anchors)
        gates = np.nan_to_num(gates, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        gate_floor = max(min(self.config.anchor_gate_threshold, 1.0), 0.0)
        anchor_mask: Optional[np.ndarray] = None
        if anchor_scores is not None and anchor_scores.size:
            scaled_scores = anchor_scores.copy()
            max_score = float(np.max(scaled_scores))
            if max_score > 0:
                scaled_scores /= max_score
            anchor_mask = scaled_scores >= gate_floor
            if self.config.strict_anchor_filter and np.any(anchor_mask):
                gates = np.where(anchor_mask, gates * np.maximum(scaled_scores, gate_floor), 0.0)
            else:
                gates = gates * np.maximum(scaled_scores, gate_floor)
            if anchor_mask is not None and np.any(anchor_mask):
                topic_mask = np.asarray(anchor_mask, dtype=bool)
                if self.adjacency is not None and self.adjacency.size:
                    neighbour_mask = np.any(self.adjacency[np.where(anchor_mask)[0]] > 0, axis=0)
                    topic_mask = np.logical_or(topic_mask, neighbour_mask)
            else:
                topic_mask = np.logical_or(topic_mask, anchor_mask)
        gates = np.where(gates > 0.0, gates, np.full_like(gates, 1e-6))
        collaboration_matrix = self._compute_collaboration_matrix(
            similarity,
            gates,
            topic_mask,
        )
        cohesion_matrix = self._cohesion_matrix(similarity)
        cohesion_degrees = cohesion_matrix.sum(axis=1)
        lambda_cov, lambda_coh = self._knowledge_weights()
        gated_relevance = gates * smoothed
        feature_map = self._dpp_feature_map(similarity)
        _, group_members = self._build_group_membership(topic_mask)
        group_bounds = self._resolve_group_bounds(size, group_members)
        (
            indices,
            coverage_state,
            coverage_average,
            diversity_value,
            cohesion_numerator,
            cohesion_denominator,
            collaboration_value,
        ) = self._greedy_select(
            gated_relevance,
            similarity,
            size,
            topic_mask=topic_mask,
            collaboration_matrix=collaboration_matrix,
            lambda_cov=lambda_cov,
            lambda_coh=lambda_coh,
            cohesion_matrix=cohesion_matrix,
            cohesion_degrees=cohesion_degrees,
            feature_map=feature_map,
            group_members=group_members,
            group_bounds=group_bounds,
        )
        concepts = tuple(self.index_to_entity[idx] for idx in indices)
        cohesion = self._cohesion_value(cohesion_numerator, cohesion_denominator)
        collaboration = float(collaboration_value)
        coverage_total = float(coverage_average)
        knowledge_raw = lambda_cov * coverage_total + lambda_coh * cohesion
        relevance_total = float(gated_relevance[indices].sum()) if len(indices) else 0.0
        diversity_total = float(diversity_value)
        gate_mean = float(gates[indices].mean()) if len(indices) else 0.0
        return KnowledgeSelection(
            concepts=concepts,
            relevance=relevance_total,
            coverage=coverage_total,
            cohesion=float(cohesion),
            collaboration=collaboration,
            diversity=diversity_total,
            knowledge_raw=float(knowledge_raw),
            gate_mean=gate_mean,
        )

    # Internal utilities -----------------------------------------------------------
    def _build_index_lookup(self) -> None:
        size = len(self.entities)
        self.index_to_entity = [""] * size
        for name, index in self.entities.items():
            if 0 <= index < size:
                self.index_to_entity[index] = name
        self._entity_token_cache = {}

    def _build_graph(self, triples: Sequence[KnowledgeEdge]) -> None:
        count = len(self.entities)
        if count == 0:
            self.adjacency = None
            self.degree = None
            self.graph_laplacian = None
            self.transition = None
            return
        counts = np.zeros((count, count), dtype=float)
        for edge in triples:
            head_idx = self.entities[edge.head]
            tail_idx = self.entities[edge.tail]
            counts[head_idx, tail_idx] += 1.0
            counts[tail_idx, head_idx] += 1.0
        total = float(counts.sum())
        if total <= 0.0:
            adjacency = np.zeros_like(counts)
        else:
            marginals = counts.sum(axis=1)
            probabilities = marginals / total
            gamma = max(self.config.smoothing_gamma, 0.0)
            if gamma != 1.0:
                smoothed = probabilities**gamma
                normaliser = float(smoothed.sum()) or 1.0
                probabilities = smoothed / normaliser
            adjacency = np.zeros_like(counts)
            shift = math.log(max(self.config.sppmi_shift, 1.0))
            rows, cols = np.nonzero(counts)
            for i, j in zip(rows.tolist(), cols.tolist()):
                p_ij = counts[i, j] / total
                p_i = max(probabilities[i], 1e-12)
                p_j = max(probabilities[j], 1e-12)
                pmi = math.log(max(p_ij / (p_i * p_j + 1e-12), 1e-12))
                value = max(pmi - shift, 0.0)
                if value == 0.0 and counts[i, j] > 0:
                    value = 1e-6
                adjacency[i, j] = value
        np.fill_diagonal(adjacency, 0.0)
        self.adjacency = adjacency
        degrees = adjacency.sum(axis=1)
        self.degree = degrees
        self.graph_laplacian = np.diag(degrees) - adjacency
        if np.all(degrees == 0):
            self.transition = np.zeros_like(adjacency)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                transition = np.divide(
                    adjacency,
                    degrees[:, None],
                    where=degrees[:, None] > 0,
                )
            transition[np.isnan(transition)] = 0.0
            self.transition = transition

    def _compute_similarity_matrix(self) -> Optional[FloatArray]:
        if self.embeddings is None:
            if self.adjacency is None:
                return None
            max_value = float(np.max(self.adjacency)) if self.adjacency.size else 0.0
            if max_value <= 0:
                return cast(FloatArray, np.zeros_like(self.adjacency))
            scaled = self.adjacency / max_value
            return cast(FloatArray, np.asarray(scaled, dtype=float))
        matrix = np.asarray(self.embeddings, dtype=float)
        if matrix.size == 0:
            return None
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        similarity = matrix @ matrix.T
        similarity = similarity / norms
        similarity = similarity / norms.T
        similarity = np.clip(similarity, 0.0, 1.0)
        np.fill_diagonal(similarity, 1.0)
        return cast(FloatArray, np.asarray(similarity, dtype=float))

    @staticmethod
    def _normalise_anchor_token(token: str) -> str:
        return token.replace("_", " ").replace("-", " ")

    def _prepare_anchor_tokens(self, tokens: Sequence[str]) -> set[str]:
        anchors: set[str] = set()
        for token in tokens:
            if token is None:
                continue
            expanded = self._normalise_anchor_token(str(token))
            for piece in tokenize(expanded):
                piece = piece.strip()
                if piece:
                    anchors.add(piece)
        return anchors

    def _concept_tokens(self, name: str) -> set[str]:
        if name in self._entity_token_cache:
            return self._entity_token_cache[name]
        expanded = self._normalise_anchor_token(name)
        tokens = {piece for piece in tokenize(expanded) if piece}
        self._entity_token_cache[name] = tokens
        return tokens

    def _anchor_overlap_scores(self, anchors: set[str]) -> np.ndarray:
        scores = np.zeros(len(self.index_to_entity), dtype=float)
        if not anchors or not self.index_to_entity:
            return scores
        anchor_count = max(len(anchors), 1)
        for idx, name in enumerate(self.index_to_entity):
            concept_tokens = self._concept_tokens(name)
            overlap = anchors.intersection(concept_tokens)
            if overlap:
                scores[idx] = len(overlap) / anchor_count
        return scores

    def _align_prompt_vector(self, vector: np.ndarray) -> FloatArray:
        target_dim = self.embeddings.shape[1] if self.embeddings is not None else vector.size
        if vector.size == target_dim:
            return cast(FloatArray, np.asarray(vector, dtype=float))
        if vector.size > target_dim:
            return cast(FloatArray, np.asarray(vector[:target_dim], dtype=float))
        padded = np.zeros(target_dim, dtype=float)
        padded[: vector.size] = vector
        return cast(FloatArray, padded)

    def _compute_relevance(self, prompt_vector: np.ndarray) -> FloatArray:
        if self.embeddings is None:
            return cast(FloatArray, np.zeros(len(self.entities), dtype=float))
        prompt_norm = float(np.linalg.norm(prompt_vector))
        if prompt_norm == 0.0:
            return cast(FloatArray, np.zeros(len(self.entities), dtype=float))
        entity_norms = np.linalg.norm(self.embeddings, axis=1)
        denom = entity_norms * prompt_norm
        denom = np.where(denom == 0, 1.0, denom)
        relevance = (self.embeddings @ prompt_vector) / denom
        clipped = np.clip(relevance, -1.0, 1.0)
        return cast(FloatArray, np.asarray(clipped, dtype=float))

    def _smoothed_relevance(self, relevance: np.ndarray) -> FloatArray:
        if (
            self.graph_laplacian is None
            or self.graph_laplacian.size == 0
            or self.config.smoothing_lambda <= 0
        ):
            return cast(FloatArray, np.asarray(relevance, dtype=float))
        n = relevance.shape[0]
        identity = np.eye(n)
        matrix = identity + self.config.smoothing_lambda * self.graph_laplacian
        try:
            smoothed = np.linalg.solve(matrix, relevance)
        except np.linalg.LinAlgError:
            smoothed = np.linalg.pinv(matrix) @ relevance
        return cast(FloatArray, np.asarray(smoothed, dtype=float))

    def _topic_weights(self, relevance: np.ndarray) -> np.ndarray:
        scaled = 0.5 * (np.asarray(relevance, dtype=float) + 1.0)
        return np.clip(scaled, 0.0, 1.0)

    def _topic_mask(self, topic_weights: np.ndarray) -> np.ndarray:
        mask = topic_weights >= float(self.config.topic_threshold)
        if not np.any(mask) and topic_weights.size:
            order = np.argsort(topic_weights)[::-1]
            top = order[: max(1, min(self.config.anchor_pool, topic_weights.size))]
            mask = np.zeros_like(topic_weights, dtype=bool)
            mask[top] = True
        return mask

    def _select_anchors(
        self,
        relevance: np.ndarray,
        topic_mask: np.ndarray,
        selection_size: int,
    ) -> np.ndarray:
        candidates = np.where(topic_mask)[0]
        if candidates.size == 0:
            candidates = np.arange(relevance.size)
        multiplier = max(self.config.anchor_multiplier, 1.0)
        dynamic_size = int(math.ceil(multiplier * max(1, selection_size)))
        if self.config.anchor_pool > 0:
            dynamic_size = min(dynamic_size, int(self.config.anchor_pool))
        pool_size = max(1, min(dynamic_size, candidates.size))
        order = np.argsort(relevance[candidates])[::-1]
        anchors = candidates[order[:pool_size]]
        if anchors.size == 0 and relevance.size:
            anchors = np.array([int(np.argmax(relevance))], dtype=int)
        return np.asarray(anchors, dtype=int)

    def _compute_anchor_gates(
        self,
        similarity: np.ndarray,
        anchors: np.ndarray,
    ) -> tuple[FloatArray, FloatArray]:
        n = similarity.shape[0]
        if n == 0:
            zero = cast(FloatArray, np.zeros(0, dtype=float))
            return zero, zero
        if anchors.size == 0:
            ones = cast(FloatArray, np.ones(n, dtype=float))
            return ones, cast(FloatArray, np.ones(n, dtype=float))
        matrix = np.asarray(similarity, dtype=float)
        with np.errstate(invalid="ignore"):
            row_sums = matrix.sum(axis=1)
        row_sums = np.where(row_sums <= 0.0, 1.0, row_sums)
        with np.errstate(divide="ignore", invalid="ignore"):
            transition = np.divide(matrix, row_sums[:, None], where=row_sums[:, None] > 0)
        transition[np.isnan(transition)] = 0.0
        alpha = float(self.config.ppr_alpha)
        identity = np.eye(n)
        system = identity - alpha * transition
        try:
            fundamental = np.linalg.inv(system)
        except np.linalg.LinAlgError:
            fundamental = np.linalg.pinv(system)
        fundamental = np.asarray(fundamental, dtype=float)
        rho = np.zeros(n, dtype=float)
        for anchor in anchors.tolist():
            column = (1.0 - alpha) * fundamental[:, anchor]
            rho = np.maximum(rho, np.asarray(column, dtype=float))
        rho = np.clip(rho, 0.0, 1.0)
        tau_g = max(float(self.config.gate_bias), 1e-8)
        gates = rho / (rho + tau_g)
        clipped = np.clip(gates, 0.0, 1.0)
        return (
            cast(FloatArray, np.asarray(clipped, dtype=float)),
            cast(FloatArray, np.asarray(rho, dtype=float)),
        )

    def _compute_collaboration_matrix(
        self,
        similarity: np.ndarray,
        gates: np.ndarray,
        topic_mask: np.ndarray,
    ) -> FloatArray:
        n = similarity.shape[0]
        matrix = np.zeros((n, n), dtype=float)
        off_indices = np.where(~topic_mask)[0]
        on_indices = np.where(topic_mask)[0]
        if off_indices.size == 0 or on_indices.size == 0:
            return cast(FloatArray, matrix)
        for off_idx in off_indices.tolist():
            for on_idx in on_indices.tolist():
                value = float(gates[off_idx]) * float(similarity[off_idx, on_idx])
                if value > 0.0:
                    matrix[off_idx, on_idx] = value
        return cast(FloatArray, matrix)

    @staticmethod
    def _cohesion_matrix(similarity: np.ndarray) -> FloatArray:
        matrix = np.asarray(similarity, dtype=float).copy()
        np.fill_diagonal(matrix, 0.0)
        return cast(FloatArray, matrix)

    @staticmethod
    def _cohesion_value(numerator: float, denominator: float) -> float:
        if denominator <= 0:
            return 0.0
        return numerator / denominator

    def _dpp_feature_map(self, similarity: np.ndarray) -> FloatArray:
        matrix = np.asarray(similarity, dtype=float)
        if matrix.size == 0:
            return cast(FloatArray, np.zeros((0, 0), dtype=float))
        sym = 0.5 * (matrix + matrix.T)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(sym)
        except np.linalg.LinAlgError:
            eigenvalues, eigenvectors = np.linalg.eig(sym)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        floor = max(float(self.config.dpp_eigen_floor), 0.0)
        mask = eigenvalues > floor
        if not np.any(mask):
            return cast(FloatArray, np.zeros((sym.shape[0], 0), dtype=float))
        selected = eigenvalues[mask]
        scaled = np.sqrt(selected)
        features = eigenvectors[:, mask] * scaled
        return cast(FloatArray, np.asarray(features, dtype=float))

    @staticmethod
    def _cholesky_rank_one_update(cholesky: np.ndarray, vector: np.ndarray) -> FloatArray:
        if vector.size == 0:
            return cast(FloatArray, np.asarray(cholesky, dtype=float))
        L = np.asarray(cholesky, dtype=float).copy()
        w = np.asarray(vector, dtype=float).copy()
        n = L.shape[0]
        for k in range(n):
            diag = L[k, k]
            r = math.hypot(diag, w[k])
            if r == 0.0:
                continue
            c = r / diag if abs(diag) > 1e-12 else 1.0
            s = w[k] / diag if abs(diag) > 1e-12 else 0.0
            L[k, k] = r
            if k + 1 < n:
                L[k + 1 :, k] = (L[k + 1 :, k] + s * w[k + 1 :]) / c
                w[k + 1 :] = c * w[k + 1 :] - s * L[k + 1 :, k]
        return cast(FloatArray, L)

    def _knowledge_weights(self) -> tuple[float, float]:
        weights = np.asarray(
            [
                max(self.config.coverage_mix, 0.0),
                max(self.config.cohesion_mix, 0.0),
            ],
            dtype=float,
        )
        total = float(weights.sum())
        if total <= 0:
            return 1.0, 0.0
        normalised = weights / total
        return float(normalised[0]), float(normalised[1])

    def _build_group_membership(
        self,
        topic_mask: np.ndarray,
    ) -> tuple[dict[int, set[str]], dict[str, set[int]]]:
        membership: dict[int, set[str]] = {}
        for idx, name in enumerate(self.index_to_entity):
            groups = self._concept_groups.get(name, set())
            membership[idx] = set(groups)
        on_indices = np.where(topic_mask)[0]
        off_indices = np.where(~topic_mask)[0]
        for idx in on_indices.tolist():
            membership.setdefault(idx, set()).add("on-topic")
        for idx in off_indices.tolist():
            membership.setdefault(idx, set()).add("off-topic")
        group_members: dict[str, set[int]] = {}
        for idx, groups in membership.items():
            for group in groups:
                group_members.setdefault(group, set()).add(idx)
        return membership, group_members

    @staticmethod
    def _interpret_bound(
        value: Optional[float],
        selection_size: int,
        *,
        is_lower: bool,
    ) -> Optional[int]:
        if value is None:
            return None
        if not math.isfinite(value):
            return None if not is_lower else 0
        if value < 0:
            return 0 if is_lower else None
        if 0 < value < 1:
            scaled = value * selection_size
            return int(math.ceil(scaled)) if is_lower else int(math.floor(scaled))
        if value == 0:
            return 0
        rounded = int(math.ceil(value)) if is_lower else int(math.floor(value))
        return max(rounded, 0)

    def _resolve_group_bounds(
        self,
        selection_size: int,
        group_members: dict[str, set[int]],
    ) -> dict[str, tuple[int, int]]:
        bounds: dict[str, tuple[int, int]] = {}
        on_count = len(group_members.get("on-topic", set()))
        off_count = len(group_members.get("off-topic", set()))
        on_lower = math.ceil(max(self.config.on_topic_min_ratio, 0.0) * selection_size)
        on_lower = min(on_lower, on_count, selection_size)
        off_lower = math.ceil(max(self.config.off_topic_min_ratio, 0.0) * selection_size)
        off_lower = min(off_lower, off_count, selection_size)
        if on_lower + off_lower > selection_size:
            excess = on_lower + off_lower - selection_size
            off_lower = max(0, off_lower - excess)
        off_upper_ratio = max(self.config.off_topic_max_ratio, 0.0)
        off_upper = math.floor(off_upper_ratio * selection_size)
        if off_upper <= 0 and off_lower > 0:
            off_upper = off_lower
        off_upper = min(max(off_upper, off_lower), off_count, selection_size)
        bounds["on-topic"] = (on_lower, max(on_lower, min(selection_size, on_count)))
        bounds["off-topic"] = (off_lower, off_upper)
        for name, members in group_members.items():
            if name in {"on-topic", "off-topic"}:
                continue
            available = len(members)
            lower_raw, upper_raw = self._group_bounds.get(name, (None, None))
            lower = self._interpret_bound(lower_raw, selection_size, is_lower=True)
            upper = self._interpret_bound(upper_raw, selection_size, is_lower=False)
            if lower is None:
                lower = 0
            lower = min(lower, available, selection_size)
            if upper is None or upper <= 0:
                upper = min(available, selection_size)
            else:
                upper = min(max(upper, lower), available, selection_size)
            bounds[name] = (lower, upper)
        for name, (lower_raw, upper_raw) in self._group_bounds.items():
            if name in bounds:
                continue
            lower = self._interpret_bound(lower_raw, selection_size, is_lower=True)
            upper = self._interpret_bound(upper_raw, selection_size, is_lower=False)
            if lower is None:
                lower = 0
            available = len(group_members.get(name, set()))
            lower = min(lower, available, selection_size)
            if upper is None or upper <= 0:
                upper = min(available, selection_size)
            else:
                upper = min(max(upper, lower), available, selection_size)
            bounds[name] = (lower, upper)
        return bounds

    def _greedy_select(
        self,
        gated_relevance: np.ndarray,
        similarity: np.ndarray,
        top_k: int,
        *,
        topic_mask: np.ndarray,
        collaboration_matrix: np.ndarray,
        lambda_cov: float,
        lambda_coh: float,
        cohesion_matrix: np.ndarray,
        cohesion_degrees: np.ndarray,
        feature_map: np.ndarray,
        group_members: dict[str, set[int]],
        group_bounds: dict[str, tuple[int, int]],
    ) -> tuple[np.ndarray, np.ndarray, float, float, float, float, float]:
        n = gated_relevance.shape[0]
        coverage_state = np.zeros(n, dtype=float)
        coverage_sum = 0.0
        selected: list[int] = []
        available = set(range(n))
        mu = float(self.config.knowledge_strength)
        gamma = float(self.config.collaboration_strength)
        tau = float(self.config.diversity_strength)
        cohesion_numerator = 0.0
        cohesion_denominator = 0.0
        collaboration_total = 0.0
        collab_map: dict[int, float] = {}
        selected_on: list[int] = []
        selected_off: list[int] = []
        feature_dim = feature_map.shape[1] if feature_map.ndim == 2 else 0
        chol = np.eye(feature_dim, dtype=float) if feature_dim else np.eye(0, dtype=float)
        logdet_value = 0.0
        selected_counts = {group: 0 for group in group_bounds}
        remaining_counts = {group: len(group_members.get(group, set())) for group in group_bounds}

        def is_feasible(idx: int) -> bool:
            slots_after = top_k - (len(selected) + 1)
            for group, (lower, upper) in group_bounds.items():
                members = group_members.get(group, set())
                in_group = idx in members
                count = selected_counts.get(group, 0)
                if in_group and count + 1 > upper:
                    return False
                count_after = count + (1 if in_group else 0)
                remaining = remaining_counts.get(group, 0) - (1 if in_group else 0)
                achievable = count_after + min(remaining, max(slots_after, 0))
                if achievable < lower:
                    return False
            return True

        while len(selected) < top_k and available:
            best_idx: Optional[int] = None
            best_payload: Optional[_GreedyCandidate] = None
            best_score = float("-inf")
            for idx in list(available):
                if not is_feasible(idx):
                    continue
                delta_rel = float(gated_relevance[idx])
                if delta_rel <= 1e-4 and not topic_mask[idx]:
                    continue
                candidate_cov = np.maximum(coverage_state, similarity[:, idx])
                candidate_cov_sum = float(candidate_cov.sum())
                delta_cov = (candidate_cov_sum - coverage_sum) / max(n, 1)
                connection = float(cohesion_matrix[idx, selected].sum()) if selected else 0.0
                candidate_cohesion_num = cohesion_numerator + 2.0 * connection
                candidate_cohesion_den = cohesion_denominator + float(cohesion_degrees[idx])
                current_cohesion = (
                    cohesion_numerator / cohesion_denominator if cohesion_denominator > 0 else 0.0
                )
                new_cohesion = (
                    candidate_cohesion_num / candidate_cohesion_den
                    if candidate_cohesion_den > 0
                    else 0.0
                )
                delta_coh = new_cohesion - current_cohesion
                candidate_collab_map = dict(collab_map)
                delta_collab = 0.0
                if topic_mask[idx]:
                    for off_idx in selected_off:
                        current_best = candidate_collab_map.get(off_idx, 0.0)
                        bridge = float(collaboration_matrix[off_idx, idx])
                        if bridge > current_best:
                            delta_collab += bridge - current_best
                            candidate_collab_map[off_idx] = bridge
                else:
                    best_bridge = 0.0
                    for on_idx in selected_on:
                        bridge = float(collaboration_matrix[idx, on_idx])
                        if bridge > best_bridge:
                            best_bridge = bridge
                    candidate_collab_map[idx] = best_bridge
                    delta_collab += best_bridge
                knowledge_delta = lambda_cov * delta_cov + lambda_coh * delta_coh
                feature_vector = (
                    np.asarray(feature_map[idx], dtype=float)
                    if feature_dim
                    else np.zeros(0, dtype=float)
                )
                if feature_dim:
                    try:
                        solution = np.linalg.solve(chol, feature_vector)
                    except np.linalg.LinAlgError:
                        solution = np.linalg.lstsq(chol, feature_vector, rcond=None)[0]
                    delta_div = math.log1p(float(solution @ solution))
                else:
                    delta_div = 0.0
                score = delta_rel + mu * knowledge_delta + gamma * delta_collab + tau * delta_div
                if score > best_score:
                    best_score = score
                    best_idx = idx
                    best_payload = _GreedyCandidate(
                        coverage=cast(FloatArray, np.asarray(candidate_cov, dtype=float)),
                        coverage_sum=float(candidate_cov_sum),
                        cohesion_num=float(candidate_cohesion_num),
                        cohesion_den=float(candidate_cohesion_den),
                        collab_map=dict(candidate_collab_map),
                        delta_collab=float(delta_collab),
                        feature_vector=cast(FloatArray, np.asarray(feature_vector, dtype=float)),
                        delta_div=float(delta_div),
                    )
            if best_idx is None or best_payload is None:
                break
            idx = best_idx
            selected.append(idx)
            available.remove(idx)
            coverage_state = np.asarray(best_payload.coverage, dtype=float)
            coverage_sum = float(best_payload.coverage_sum)
            cohesion_numerator = float(best_payload.cohesion_num)
            cohesion_denominator = float(best_payload.cohesion_den)
            collab_map = dict(best_payload.collab_map)
            if topic_mask[idx]:
                selected_on.append(idx)
            else:
                selected_off.append(idx)
            collaboration_total += float(best_payload.delta_collab)
            if feature_dim:
                feature_vector = np.asarray(best_payload.feature_vector, dtype=float)
                chol = self._cholesky_rank_one_update(chol, feature_vector)
                logdet_value += float(best_payload.delta_div)
            for group in group_bounds:
                if idx in group_members.get(group, set()):
                    selected_counts[group] = selected_counts.get(group, 0) + 1
                    remaining_counts[group] = max(0, remaining_counts.get(group, 0) - 1)
        if not selected:
            return (
                np.asarray([], dtype=int),
                cast(FloatArray, np.asarray(coverage_state, dtype=float)),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
        average_coverage = float(coverage_state.sum() / max(n, 1))
        return (
            np.asarray(selected, dtype=int),
            cast(FloatArray, np.asarray(coverage_state, dtype=float)),
            average_coverage,
            logdet_value,
            cohesion_numerator,
            cohesion_denominator,
            collaboration_total,
        )

    def _compute_cohesion(self, indices: Sequence[int]) -> float:
        # Deprecated shim retained for compatibility with historical interfaces.
        if self.similarity is None:
            return 0.0
        matrix = self._cohesion_matrix(self.similarity)
        degrees = matrix.sum(axis=1)
        numerator = 0.0
        denominator = 0.0
        subset = np.asarray(indices, dtype=int)
        if subset.size == 0:
            return 0.0
        numerator = float(matrix[np.ix_(subset, subset)].sum())
        denominator = float(degrees[subset].sum())
        return self._cohesion_value(numerator, denominator)

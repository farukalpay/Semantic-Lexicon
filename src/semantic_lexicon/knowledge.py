"""Knowledge network management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .config import KnowledgeConfig
from .logging import configure_logging

LOGGER = configure_logging(logger_name=__name__)


@dataclass(frozen=True)
class KnowledgeEdge:
    head: str
    relation: str
    tail: str


class KnowledgeNetwork:
    """A light-weight knowledge graph with simple scoring functions."""

    def __init__(self, config: KnowledgeConfig | None = None) -> None:
        self.config = config or KnowledgeConfig()
        self.entities: Dict[str, int] = {}
        self.relations: Dict[str, int] = {}
        self.embeddings: np.ndarray | None = None
        self.relation_matrices: np.ndarray | None = None

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
        self.embeddings = rng.normal(0, 0.1, size=(entity_dim, self.config.max_relations))
        self.relation_matrices = rng.normal(0, 0.1, size=(relation_dim, self.config.max_relations, self.config.max_relations))
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
                total_loss += error ** 2
                grad_head = -2 * error * relation_matrix @ tail_vec
                grad_tail = -2 * error * relation_matrix.T @ head_vec
                grad_rel = -2 * error * np.outer(head_vec, tail_vec)
                self.embeddings[head_idx] -= learning_rate * grad_head
                self.embeddings[tail_idx] -= learning_rate * grad_tail
                self.relation_matrices[relation_idx] -= learning_rate * grad_rel
            LOGGER.debug("Knowledge epoch %s | loss=%.4f", epoch + 1, total_loss / len(triples))
        LOGGER.info("Trained knowledge network with %d entities", entity_dim)

    # Querying --------------------------------------------------------------------
    def neighbours(self, entity: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.embeddings is None or entity not in self.entities:
            return []
        idx = self.entities[entity]
        target = self.embeddings[idx]
        scores = []
        for name, other_idx in self.entities.items():
            if other_idx == idx:
                continue
            score = float(np.dot(target, self.embeddings[other_idx]))
            scores.append((name, score))
        return sorted(scores, key=lambda item: item[1], reverse=True)[:top_k]

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

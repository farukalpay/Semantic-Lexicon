"""Knowledge network representations and utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class KnowledgeEntry:
    concept: str
    description: str
    related: List[str] = field(default_factory=list)


class KnowledgeNetwork:
    """Simple co-occurrence based knowledge network."""

    def __init__(self) -> None:
        self.nodes: Dict[str, KnowledgeEntry] = {}
        self.edge_weights: Dict[Tuple[str, str], float] = {}

    def add_entry(self, entry: KnowledgeEntry) -> None:
        LOGGER.debug("Adding knowledge entry for concept %s", entry.concept)
        self.nodes[entry.concept] = entry
        for related in entry.related:
            key = tuple(sorted((entry.concept, related)))
            self.edge_weights[key] = self.edge_weights.get(key, 0.0) + 1.0

    def get_entry(self, concept: str) -> KnowledgeEntry:
        return self.nodes[concept]

    def related_concepts(self, concept: str, *, top_k: int = 3) -> List[str]:
        scores: List[Tuple[str, float]] = []
        for other in self.nodes:
            if other == concept:
                continue
            key = tuple(sorted((concept, other)))
            weight = self.edge_weights.get(key, 0.0)
            if weight > 0:
                scores.append((other, weight))
        scores.sort(key=lambda item: item[1], reverse=True)
        return [concept for concept, _ in scores[:top_k]]

    def to_dict(self) -> Dict[str, Dict[str, List[str]]]:
        return {
            "nodes": {concept: {"description": entry.description, "related": entry.related} for concept, entry in self.nodes.items()},
            "edges": {"|".join(key): weight for key, weight in self.edge_weights.items()},
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Dict[str, List[str]]]) -> "KnowledgeNetwork":
        network = cls()
        for concept, entry in payload.get("nodes", {}).items():
            network.add_entry(KnowledgeEntry(concept=concept, description=entry.get("description", ""), related=entry.get("related", [])))
        network.edge_weights = {}
        for key, weight in payload.get("edges", {}).items():
            a, b = key.split("|")
            network.edge_weights[tuple(sorted((a, b)))] = float(weight)
        return network

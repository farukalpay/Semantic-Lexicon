# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from .oracle import Oracle, OracleReport

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from .graph_api import GraphAPI


def _entity_aliases_all(graph: GraphAPI, entity_id: str) -> Iterable[str]:
    entity = graph.get_entity(entity_id)
    if entity is None:
        return
    yield entity.label
    yield from entity.aliases


def _resolve_subject(graph: GraphAPI, words_lc: list[str]) -> Optional[str]:
    max_back = min(24, len(words_lc))
    for start in range(len(words_lc) - 1, max(-1, len(words_lc) - max_back) - 1, -1):
        for length in (3, 2, 1):
            begin = start - length + 1
            if begin < 0:
                continue
            surface = " ".join(words_lc[begin : start + 1])
            entity_id = graph.find_entity_by_surface(surface)
            if entity_id:
                return entity_id
    return None


def _match_relation_suffix(
    rel_lex: dict[tuple[str, ...], str], words_lc: list[str]
) -> Optional[str]:
    max_len = max((len(key) for key in rel_lex), default=0)
    for length in range(min(max_len, len(words_lc)), 0, -1):
        candidate = tuple(words_lc[-length:])
        if candidate in rel_lex:
            return rel_lex[candidate]
    return None


class GraphKBOracle(Oracle):
    """Truth-aware decoding oracle backed by a GraphAPI implementation."""

    def __init__(
        self,
        graph: GraphAPI,
        relation_lexicon: dict[tuple[str, ...], str],
        alias_to_token_ids: dict[str, list[int]],
    ) -> None:
        self.graph = graph
        self.rel_lex = relation_lexicon
        self.alias_to_token_ids = alias_to_token_ids
        self._cache: dict[tuple[str, str], set[int]] = {}

    def evaluate(
        self, prefix_token_ids: Sequence[int], next_logits: np.ndarray, vocab: Sequence[str]
    ) -> OracleReport:
        vocab_size = len(vocab)
        safe = np.ones(vocab_size, dtype=bool)
        reasons: list[set[str]] = [set() for _ in range(vocab_size)]

        words = [vocab[token_id] for token_id in prefix_token_ids]
        words_lc = [word.lower() for word in words]

        subject_id = _resolve_subject(self.graph, words_lc)
        relation = _match_relation_suffix(self.rel_lex, words_lc)

        if subject_id and relation:
            allowed_ids = self._allowed_token_ids(subject_id, relation)
            if allowed_ids:
                for token_id in range(vocab_size):
                    if token_id not in allowed_ids:
                        token = vocab[token_id]
                        if token and token[0].isalpha() and token[0].isupper():
                            safe[token_id] = False
                            reasons[token_id].add(
                                f"graph:contradiction:{self.graph.label(subject_id)}_{relation}"
                            )
                    else:
                        reasons[token_id].add("graph:required_object")
        return OracleReport(safe_mask=safe, reasons=reasons)

    def _allowed_token_ids(self, subject_id: str, relation: str) -> set[int]:
        cache_key = (subject_id, relation)
        if cache_key in self._cache:
            return self._cache[cache_key]
        allowed: set[int] = set()
        for object_id in self.graph.objects(subject_id, relation):
            for alias in _entity_aliases_all(self.graph, object_id):
                for token_id in self.alias_to_token_ids.get(alias.lower(), []):
                    allowed.add(token_id)
        self._cache[cache_key] = allowed
        return allowed

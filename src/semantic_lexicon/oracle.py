# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


class SafetyVerdict(Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


@dataclass
class OracleReport:
    """
    Result of calling an Oracle at a single decode step.

    Attributes
    ----------
    safe_mask : np.ndarray[bool] shape [V]
        True for tokens allowed by the oracle.
    reasons : list[set[str]] length V
        For each token id, a (possibly empty) set of human-readable
        reasons/labels explaining blocks or passes (for diagnostics).
    """

    safe_mask: np.ndarray
    reasons: list[set[str]]


class Oracle(Protocol):
    """
    An Oracle inspects the current prefix and next-token logits and decides,
    for every vocabulary token, whether it is SAFE to emit now.

    Implementations should be **pure** and **stateless** at call time.
    """

    def evaluate(
        self,
        prefix_token_ids: Sequence[int],
        next_logits: np.ndarray,  # shape [V]
        vocab: Sequence[str],
    ) -> OracleReport: ...


class NullOracle:
    """Allows every token; useful for A/B tests."""

    def evaluate(
        self, prefix_token_ids: Sequence[int], next_logits: np.ndarray, vocab: Sequence[str]
    ) -> OracleReport:
        V = len(vocab)
        return OracleReport(safe_mask=np.ones(V, dtype=bool), reasons=[set() for _ in range(V)])


class CompositeOracle:
    """
    Accepts a token iff **all** sub-oracles accept it (logical AND).
    Reasons are unioned for transparency.
    """

    def __init__(self, oracles: Sequence[Oracle]) -> None:
        self._oracles = list(oracles)

    def evaluate(
        self, prefix_token_ids: Sequence[int], next_logits: np.ndarray, vocab: Sequence[str]
    ) -> OracleReport:
        base = None
        for oracle in self._oracles:
            report = oracle.evaluate(prefix_token_ids, next_logits, vocab)
            if base is None:
                base = report
            else:
                base.safe_mask &= report.safe_mask
                for i in range(len(base.reasons)):
                    base.reasons[i] |= report.reasons[i]
        assert base is not None
        return base


class KBOracle(Oracle):
    """
    A toy, readable oracle that enforces a (subject, relation, object) fact.

    It watches for the surface pattern "… is the capital of <OBJ>"
    and, if a known SUBJECT was mentioned earlier in the prefix,
    it only allows the correct <OBJ> according to the KB.

    KB format: dict[(subject:str, relation:str)] -> object:str
    Example: {("Paris", "capital_of"): "France"}
    """

    def __init__(self, kb: dict[tuple[str, str], str]) -> None:
        self.kb = dict(kb)

    def evaluate(
        self, prefix_token_ids: Sequence[int], next_logits: np.ndarray, vocab: Sequence[str]
    ) -> OracleReport:
        vocab_size = len(vocab)
        reasons: list[set[str]] = [set() for _ in range(vocab_size)]
        safe = np.ones(vocab_size, dtype=bool)

        tokens = [vocab[t] for t in prefix_token_ids]
        subject = _find_subject(tokens, kb_subjects={s for (s, _r) in self.kb.keys()})
        expecting_capital_obj = _ends_with(tokens, ["capital", "of"])

        if subject and expecting_capital_obj:
            key = (subject, "capital_of")
            if key in self.kb:
                correct_obj = self.kb[key]
                for token_id, token in enumerate(vocab):
                    if token == correct_obj:
                        reasons[token_id].add("kb:required_object")
                    elif _is_candidate_object(token):
                        safe[token_id] = False
                        reasons[token_id].add(f"kb:contradiction:{subject}_capital_of_{token}")
                return OracleReport(safe_mask=safe, reasons=reasons)

        return OracleReport(safe_mask=safe, reasons=reasons)


def _ends_with(tokens: Sequence[str], suffix: Sequence[str]) -> bool:
    if len(tokens) < len(suffix):
        return False
    return [t.lower() for t in tokens[-len(suffix) :]] == [s.lower() for s in suffix]


def _find_subject(tokens: Sequence[str], kb_subjects: set[str]) -> str | None:
    """
    Super naive: return the first token in the prefix that exactly matches
    a KB subject. Good enough for unit tests and demos.
    """

    for token in tokens:
        if token in kb_subjects:
            return token
    return None


def _is_candidate_object(tok: str) -> bool:
    return tok != "" and tok[0].isupper()

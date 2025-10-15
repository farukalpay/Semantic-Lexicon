# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from semantic_lexicon.decoding_tad import (
    DecodeAbortReason,
    ModelLike,
    OracleContractError,
    TADConfig,
    truth_aware_decode,
)
from semantic_lexicon.oracle import KBOracle, NullOracle, OracleReport


class ToyModel(ModelLike):
    def __init__(self) -> None:
        self._vocab = [
            "<BOS>",
            "<EOS>",
            "<ABSTAIN>",
            "Paris",
            "Berlin",
            "France",
            "Germany",
            "is",
            "the",
            "capital",
            "of",
            ".",
        ]
        self._id = {token: index for index, token in enumerate(self._vocab)}

    @property
    def vocab(self) -> list[str]:
        return self._vocab

    def eos_id(self) -> int:
        return self._id["<EOS>"]

    def next_logits(self, prefix_token_ids: Sequence[int]) -> np.ndarray:
        vocab_size = len(self._vocab)
        logits = np.full(vocab_size, -10.0, dtype=np.float64)
        logits[self._id["."]] = -2.0
        logits[self.eos_id()] = -1.5

        prefix = [self._vocab[token] for token in prefix_token_ids]
        if len(prefix) >= 2 and prefix[-2:] == ["capital", "of"]:
            logits[self._id["Germany"]] = 3.0
            logits[self._id["France"]] = 2.5
            logits[self._id["Berlin"]] = 1.0
            logits[self._id["Paris"]] = 0.5
            return logits

        if len(prefix) >= 1 and prefix[-1] == "the":
            logits[self._id["capital"]] = 3.0
            return logits

        if len(prefix) >= 1 and prefix[-1] == "is":
            logits[self._id["the"]] = 3.0
            return logits

        if len(prefix) <= 1:
            logits[self._id["Paris"]] = 2.0
            logits[self._id["Berlin"]] = 1.5
            logits[self._id["is"]] = 1.0
            return logits

        return logits


def ids(vocab: list[str], *words: str) -> list[int]:
    return [vocab.index(word) for word in words]


def test_tad_preserves_consistency_and_fixes_wrong_preference() -> None:
    model = ToyModel()
    kb = {("Paris", "capital_of"): "France"}
    oracle = KBOracle(kb)

    prefix = ids(model.vocab, "<BOS>", "Paris", "is", "the", "capital", "of")
    cfg = TADConfig(tau=0.0, max_new_tokens=1)

    outcome = truth_aware_decode(model, oracle, prefix_token_ids=prefix.copy(), cfg=cfg)
    generated = [model.vocab[index] for index in outcome.token_ids]

    assert generated == ["France"], f"got {generated}, expected ['France']"
    assert not outcome.abstained
    assert outcome.abort_reason is DecodeAbortReason.NONE
    assert any("kb:required_object" in reason for reason in outcome.logs[0].reasons_for_picked)


def test_tad_abstains_when_safe_mass_low() -> None:
    model = ToyModel()
    kb = {("Paris", "capital_of"): "France"}
    oracle = KBOracle(kb)

    prefix = ids(model.vocab, "<BOS>", "Paris", "is", "the", "capital", "of")
    cfg = TADConfig(tau=0.9, max_new_tokens=1, abstain_token=None)

    outcome = truth_aware_decode(model, oracle, prefix_token_ids=prefix.copy(), cfg=cfg)
    assert outcome.abstained
    assert outcome.abort_reason is DecodeAbortReason.ABSTAIN_LOW_SAFE_MASS
    assert len(outcome.token_ids) == 0


def test_tad_degrades_to_greedy_when_kb_unknown() -> None:
    model = ToyModel()
    kb: dict[tuple[str, str], str] = {}
    oracle = KBOracle(kb)

    prefix = ids(model.vocab, "<BOS>", "Paris", "is", "the", "capital", "of")
    cfg = TADConfig(tau=0.0, max_new_tokens=1)

    outcome = truth_aware_decode(model, oracle, prefix_token_ids=prefix.copy(), cfg=cfg)
    picked = model.vocab[outcome.token_ids[0]]

    assert picked == "Germany"
    assert outcome.abort_reason is DecodeAbortReason.NONE


def test_oracle_mask_shape_guard() -> None:
    model = ToyModel()

    class BadOracle:
        def evaluate(self, prefix_token_ids, next_logits, vocab):
            mask = np.ones((2, 2), dtype=bool)
            return OracleReport(safe_mask=mask, reasons=[set()])

    cfg = TADConfig(max_new_tokens=1, tau=0.0)
    with pytest.raises(OracleContractError):
        truth_aware_decode(model, BadOracle(), prefix_token_ids=ids(model.vocab, "<BOS>"), cfg=cfg)


def test_eos_escape_even_when_oracle_blocks_all() -> None:
    model = ToyModel()

    class BlockingOracle:
        def evaluate(self, prefix_token_ids, next_logits, vocab):
            return OracleReport(
                safe_mask=np.zeros(len(vocab), dtype=bool),
                reasons=[set() for _ in vocab],
            )

    cfg = TADConfig(max_new_tokens=3, tau=0.0, allow_eos_escape=True)
    outcome = truth_aware_decode(
        model,
        BlockingOracle(),
        prefix_token_ids=ids(model.vocab, "<BOS>"),
        cfg=cfg,
    )
    assert not outcome.abstained
    assert model.eos_id() in outcome.token_ids


def test_timeout_produces_abort_reason() -> None:
    model = ToyModel()
    oracle = NullOracle()

    calls = {"n": 0}

    def fake_now() -> float:
        calls["n"] += 1
        return float(calls["n"])

    cfg = TADConfig(max_new_tokens=50, max_decode_ms=10, tau=0.0)
    outcome = truth_aware_decode(
        model,
        oracle,
        prefix_token_ids=ids(model.vocab, "<BOS>"),
        cfg=cfg,
        now=fake_now,
    )
    assert outcome.abstained
    assert outcome.abort_reason is DecodeAbortReason.TIMEOUT


def test_sampling_within_safe_set() -> None:
    model = ToyModel()
    oracle = NullOracle()

    cfg = TADConfig(temperature=0.7, top_p=0.9, tau=0.0, max_new_tokens=3)
    rng = np.random.default_rng(0)
    outcome = truth_aware_decode(
        model,
        oracle,
        prefix_token_ids=ids(model.vocab, "<BOS>"),
        cfg=cfg,
        rng=rng,
    )
    assert len(outcome.token_ids) > 0
    assert not outcome.abstained

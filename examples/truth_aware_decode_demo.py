"""Demonstrate truth-aware decoding with a toy model and KB oracle."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic_lexicon.decoding_tad import ModelLike, TADConfig, truth_aware_decode
from semantic_lexicon.oracle import KBOracle


class ToyModel(ModelLike):
    """Word-level toy model that prefers the wrong capital by default."""

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
        self._id = {w: i for i, w in enumerate(self._vocab)}

    @property
    def vocab(self) -> Sequence[str]:
        return self._vocab

    def eos_id(self) -> int:
        return self._id["<EOS>"]

    def tokens_to_ids(self, tokens: Iterable[str]) -> List[int]:
        return [self._id[t] for t in tokens]

    def next_logits(self, prefix_token_ids: List[int]) -> np.ndarray:
        V = len(self._vocab)
        logits = np.full(V, -10.0, dtype=np.float64)

        logits[self._id["."]] = -2.0
        logits[self.eos_id()] = -1.5

        prefix = [self._vocab[t] for t in prefix_token_ids]
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


def run_demo(log_path: Path) -> dict:
    model = ToyModel()
    kb = {("Paris", "capital_of"): "France"}
    oracle = KBOracle(kb)

    prefix = model.tokens_to_ids(["<BOS>", "Paris", "is", "the", "capital", "of"])
    cfg = TADConfig(tau=0.15, max_new_tokens=4)

    outcome = truth_aware_decode(model, oracle, prefix_token_ids=prefix.copy(), cfg=cfg)
    generated_tokens = [model.vocab[i] for i in outcome.token_ids]

    log_payload = {
        "prompt_tokens": [model.vocab[i] for i in prefix],
        "generated_tokens": generated_tokens,
        "abstained": outcome.abstained,
        "steps": [
            {
                "t": step.t,
                "pi_safe": step.pi_safe,
                "picked_token": model.vocab[step.picked_id] if step.picked_id is not None else None,
                "blocked_count": step.blocked_count,
                "reasons_for_picked": step.reasons_for_picked,
                "aborted": step.aborted,
            }
            for step in outcome.logs
        ],
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log_payload, indent=2))
    return log_payload


def main() -> None:
    log_dir = Path(__file__).resolve().parent / "logs"
    log_path = log_dir / "paris_capital_truth_aware_decode.json"
    payload = run_demo(log_path)
    try:
        display_path = log_path.relative_to(ROOT)
    except ValueError:
        display_path = log_path
    print("Prompt tokens:", " ".join(payload["prompt_tokens"]))
    print("Generated tokens:", " ".join(payload["generated_tokens"]))
    print("Log written to", display_path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import numpy as np

from tadkit.core import TADLogitsProcessor, TADTrace, TruthOracle

try:  # pragma: no cover - exercised in environments with torch installed
    import torch  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - runtime optional dependency
    torch = None  # type: ignore[assignment]


class DummyTokenizer:
    def __init__(self) -> None:
        tokens = ["<eos>", "<ABSTAIN>", " Paris", " Lyon", " prompt"]
        self.vocab = {token: idx for idx, token in enumerate(tokens)}
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        self.eos_token_id = self.vocab["<eos>"]
        self.unk_token_id = -1

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002
        if text not in self.vocab:
            raise KeyError(text)
        return [self.vocab[text]]

    def decode(self, token_ids, skip_special_tokens=True):  # noqa: ARG002
        return "capital of France"

    def convert_ids_to_tokens(self, token_ids):
        return [self.id_to_token[token_id] for token_id in token_ids]

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def get_vocab(self):
        return dict(self.vocab)


def test_truth_oracle_from_rules_handles_strings():
    tokenizer = DummyTokenizer()
    oracle = TruthOracle.from_rules(
        [
            {
                "name": "capital",
                "when_any": ["capital of France"],
                "allow_strings": [" Paris"],
                "abstain_on_violation": True,
            }
        ],
        tokenizer=tokenizer,
    )
    active = oracle.active_for("capital of France")
    assert len(active) == 1
    assert tokenizer.vocab[" Paris"] in active[0].allow_token_ids


def test_logits_processor_forces_abstain_when_rule_blocked():
    tokenizer = DummyTokenizer()
    oracle = TruthOracle.from_rules(
        [
            {
                "name": "capital",
                "when_any": ["capital of France"],
                "allow_strings": [" Paris"],
                "abstain_on_violation": True,
            }
        ],
        tokenizer=tokenizer,
    )
    trace = TADTrace()
    processor = TADLogitsProcessor(oracle, tokenizer, trace=trace)
    if torch is not None:
        input_ids = torch.tensor([[tokenizer.vocab[" prompt"]]], dtype=torch.long)
        scores = torch.zeros((1, len(tokenizer.get_vocab())), dtype=torch.float32)
        scores[0, tokenizer.vocab[" Lyon"]] = 3.0
        scores[0, tokenizer.vocab[" Paris"]] = 1.0
    else:
        input_ids = np.array([[tokenizer.vocab[" prompt"]]], dtype=np.int64)
        scores = np.zeros((1, len(tokenizer.get_vocab())), dtype=np.float32)
        scores[0, tokenizer.vocab[" Lyon"]] = 3.0
        scores[0, tokenizer.vocab[" Paris"]] = 1.0
    processed = processor(input_ids, scores)
    if torch is not None:
        winner = int(torch.argmax(processed[0]).item())
    else:
        winner = int(np.argmax(processed[0]))
    assert winner == tokenizer.vocab["<ABSTAIN>"]
    actions = [event["action"] for event in trace.events]
    assert actions == ["block", "inject"]


def test_trace_to_table_shows_rows():
    trace = TADTrace()
    trace.log(step=0, token_id=1, action="pass", rule_names=["capital"], token=" Paris")
    table = trace.to_table()
    assert "Paris" in table
    assert "capital" in table

from __future__ import annotations

import random

import pytest

from personarag import BrandStyle, KnowledgeGate, PersonaPolicyEXP3
from tadkit.core import TruthOracle


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content
        self.response_metadata: dict[str, object] | None = {}


class FakeGenerationConfig:
    def __init__(self) -> None:
        self.logits_processor: list[object] = []


class FakeTokenizer:
    def __init__(self) -> None:
        self.vocab = {"<eos>": 0}
        self.eos_token_id = 0
        self.unk_token_id = -1

    def get_vocab(self):
        return dict(self.vocab)

    def convert_ids_to_tokens(self, token_ids):
        return ["<eos>" for _ in token_ids]


class FakeLLM:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.model = type("Model", (), {})()
        self.model.generation_config = FakeGenerationConfig()
        self.generation_config = None

    def invoke(self, input, config=None):  # noqa: ANN001, D401
        return FakeMessage("Paris. —Team WarmGuide")


def test_persona_policy_exp3_updates_weights():
    random.seed(0)
    personas = [
        BrandStyle(name="A", system="sys"),
        BrandStyle(name="B", system="sys"),
    ]
    policy = PersonaPolicyEXP3(personas, gamma=0.2)
    choice = policy.choose("question")
    before = list(policy.weights)
    policy.update(reward=1.0)
    assert policy.weights[policy.last_index] > before[policy.last_index]
    telemetry = policy.telemetry()
    assert pytest.approx(sum(telemetry["probs"])) == 1.0
    assert choice in personas


def test_knowledge_gate_attaches_trace_metadata():
    llm = FakeLLM()
    oracle = TruthOracle.from_rules(
        [
            {
                "name": "capitals",
                "when_any": ["capital"],
                "allow_token_ids": [0],
                "abstain_on_violation": True,
            }
        ]
    )
    gate = KnowledgeGate(llm, oracle=oracle)
    message = gate.invoke("prompt")
    assert isinstance(message, FakeMessage)
    assert "tad_trace" in message.response_metadata
    assert llm.model.generation_config.logits_processor == []

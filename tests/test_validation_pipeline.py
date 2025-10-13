# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""Tests for the cross-domain validation and performance helpers."""

from __future__ import annotations

from semantic_lexicon.analysis import (
    benchmark_inference,
    evaluate_classifier,
    load_validation_records,
)
from semantic_lexicon.config import IntentConfig
from semantic_lexicon.intent import IntentClassifier, IntentExample
from semantic_lexicon.utils import read_jsonl


def _load_training_examples() -> list[IntentExample]:
    examples: list[IntentExample] = []
    for record in read_jsonl("src/semantic_lexicon/data/intent.jsonl"):
        examples.append(
            IntentExample(
                text=str(record["text"]),
                intent=str(record["intent"]),
                feedback=0.92,
            )
        )
    return examples


def test_cross_domain_validation_metrics() -> None:
    training = _load_training_examples()
    records = load_validation_records()
    classifier = IntentClassifier()
    classifier.fit(training)
    metrics = evaluate_classifier(classifier, records)
    assert metrics.accuracy >= 0.9
    assert metrics.reward_summary["min"] >= 0.7
    assert metrics.ece_reduction >= 0.5


def test_inference_benchmark_improvement() -> None:
    training = _load_training_examples()
    records = load_validation_records()
    baseline = IntentClassifier(IntentConfig(optimized=False, cache_size=0))
    baseline.fit(training)
    optimised = IntentClassifier(IntentConfig(optimized=True, cache_size=4096))
    optimised.fit(training)
    performance = benchmark_inference(
        baseline,
        optimised,
        [record.text for record in records],
        repeat=4,
        warmup=2,
    )
    assert performance.latency_improvement_pct >= 20.0
    assert performance.optimised_latency_ms > 0.0

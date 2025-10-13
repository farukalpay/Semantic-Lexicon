# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""Run cross-domain validation and performance benchmarking."""

from __future__ import annotations

from pathlib import Path

from semantic_lexicon import (
    benchmark_inference,
    evaluate_classifier,
    load_validation_records,
    write_validation_report,
)
from semantic_lexicon.analysis.performance import PerformanceReport
from semantic_lexicon.config import IntentConfig
from semantic_lexicon.intent import IntentClassifier, IntentExample
from semantic_lexicon.utils import read_jsonl

ARCHIVE = Path("Archive")


def load_training_examples(path: Path) -> list[IntentExample]:
    examples: list[IntentExample] = []
    for record in read_jsonl(path):
        examples.append(
            IntentExample(
                text=str(record["text"]),
                intent=str(record["intent"]),
                feedback=float(record.get("feedback", 0.92)),
            )
        )
    return examples


def run_validation() -> tuple[IntentClassifier, PerformanceReport]:
    training = load_training_examples(Path("src/semantic_lexicon/data/intent.jsonl"))
    records = load_validation_records()
    optimised = IntentClassifier(IntentConfig(optimized=True, cache_size=4096))
    optimised.fit(training)
    metrics = evaluate_classifier(optimised, records)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    write_validation_report(metrics, ARCHIVE / "cross_domain_validation_report.json")
    print("Validation accuracy:", f"{metrics.accuracy:.3f}")
    print(
        "Reward summary:",
        "mean={mean:.3f} min={min:.3f} p10={p10:.3f} p90={p90:.3f}".format(**metrics.reward_summary),
    )
    print(
        "ECE:",
        f"raw={metrics.ece_before:.3f} calibrated={metrics.ece_after:.3f}"
        f" reduction={metrics.ece_reduction:.0%}",
    )

    baseline = IntentClassifier(IntentConfig(optimized=False, cache_size=0))
    baseline.fit(training)
    prompts = [record.text for record in records]
    performance = benchmark_inference(
        baseline,
        optimised,
        prompts,
        repeat=5,
        warmup=2,
    )
    with (ARCHIVE / "intent_performance_profile.json").open("w", encoding="utf8") as handle:
        import json

        json.dump(performance.to_dict(), handle, indent=2)
        handle.write("\n")
    print(
        "Latency (ms): baseline={:.3f} optimised={:.3f} improvement={:.1f}%".format(
            performance.baseline_latency_ms,
            performance.optimised_latency_ms,
            performance.latency_improvement_pct,
        )
    )
    print(
        "Memory (KB): baseline={:.1f} optimised={:.1f} reduction={:.1f}%".format(
            performance.baseline_memory_kb,
            performance.optimised_memory_kb,
            performance.memory_reduction_pct,
        )
    )
    return optimised, performance


def main() -> None:
    run_validation()


if __name__ == "__main__":
    main()

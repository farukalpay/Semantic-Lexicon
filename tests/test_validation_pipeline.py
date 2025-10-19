# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""Tests for the cross-domain validation and performance helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def test_validation_records_include_hierarchy_metadata() -> None:
    records = load_validation_records()
    energy_record = next(record for record in records if record.domain == "grid_operations")
    assert energy_record.domain_path == (
        "energy_systems",
        "grid_management",
        "grid_operations",
        "virtual_power_plant_enrollment",
    )
    assert "real_time_coordination" in energy_record.traits
    kernel_record = next(record for record in records if record.domain == "linux_kernel_operations")
    assert kernel_record.domain_path == (
        "science_and_technology",
        "platform_engineering",
        "linux_kernel_operations",
        "ubuntu_kernel_lifecycle",
    )
    assert "kernel_reliability" in kernel_record.traits


def test_cross_domain_validation_metrics() -> None:
    training = _load_training_examples()
    records = load_validation_records()
    classifier = IntentClassifier()
    classifier.fit(training)
    metrics = evaluate_classifier(classifier, records)
    assert metrics.accuracy >= 0.9
    assert metrics.reward_summary["min"] >= 0.7
    assert metrics.ece_reduction >= 0.5
    for domain in (
        "grid_operations",
        "energy_innovation",
        "grid_security",
        "storage_engineering",
        "pipeline_operations",
        "demand_response",
        "market_strategy",
        "grid_intelligence",
        "linux_kernel_operations",
    ):
        assert metrics.domain_accuracy.get(domain, 0.0) >= 0.9
    assert (
        metrics.hierarchy_accuracy.get("energy_systems > grid_management > grid_operations", 0.0)
        >= 0.9
    )
    assert (
        metrics.hierarchy_accuracy.get(
            "energy_systems > grid_management > grid_operations > virtual_power_plant_enrollment",
            0.0,
        )
        >= 0.9
    )
    assert metrics.trait_accuracy.get("real_time_coordination", 0.0) >= 0.9
    assert (
        metrics.hierarchy_accuracy.get(
            "science_and_technology > platform_engineering > linux_kernel_operations",
            0.0,
        )
        >= 0.9
    )
    ubuntu_path = (
        "science_and_technology > platform_engineering > linux_kernel_operations > "
        "ubuntu_kernel_lifecycle"
    )
    assert metrics.hierarchy_accuracy.get(ubuntu_path, 0.0) >= 0.9
    assert metrics.trait_accuracy.get("kernel_reliability", 0.0) >= 0.9
    assert metrics.trait_accuracy.get("ubuntu_support_alignment", 0.0) >= 0.9


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


def test_domain_hierarchy_duplicate_members_raise(tmp_path) -> None:
    hierarchy_path = tmp_path / "hierarchy.jsonl"
    hierarchy_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "domain": "grid_operations",
                        "path": ["energy", "grid_operations"],
                        "members": ["virtual_power_plant", "virtual_power_plant"],
                        "traits": ["coordination"],
                    }
                ),
            )
        )
    )
    with pytest.raises(ValueError, match=r"Duplicate member identifier"):
        load_validation_records(
            Path("src/semantic_lexicon/data/cross_domain_validation.jsonl"),
            hierarchy_path=hierarchy_path,
        )


def test_domain_hierarchy_contradictory_members_raise(tmp_path) -> None:
    hierarchy_path = tmp_path / "hierarchy.jsonl"
    hierarchy_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "domain": "grid_operations",
                        "path": ["energy", "grid_operations"],
                        "members": ["shared_member"],
                        "traits": ["coordination"],
                    }
                ),
                json.dumps(
                    {
                        "domain": "demand_response",
                        "path": ["energy", "demand_response"],
                        "members": ["shared_member"],
                        "traits": ["engagement"],
                    }
                ),
            )
        )
    )
    with pytest.raises(ValueError, match=r"Contradictory member 'shared_member'"):
        load_validation_records(
            Path("src/semantic_lexicon/data/cross_domain_validation.jsonl"),
            hierarchy_path=hierarchy_path,
        )


def test_domain_hierarchy_domain_casefold_conflict(tmp_path) -> None:
    hierarchy_path = tmp_path / "hierarchy.jsonl"
    hierarchy_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "domain": "grid_operations",
                        "path": ["energy_systems", "grid_operations"],
                        "members": ["virtual_power_plant"],
                        "traits": ["coordination"],
                    }
                ),
                json.dumps(
                    {
                        "domain": "Grid_Operations",
                        "path": ["energy_systems", "Grid_Operations"],
                        "members": ["telemetry_restoration"],
                        "traits": ["coordination"],
                    }
                ),
            )
        )
    )
    with pytest.raises(ValueError, match=r"conflicts with previously declared domain"):
        load_validation_records(
            Path("src/semantic_lexicon/data/cross_domain_validation.jsonl"),
            hierarchy_path=hierarchy_path,
        )


def test_domain_member_conflicts_with_domain_identifier(tmp_path) -> None:
    hierarchy_path = tmp_path / "hierarchy.jsonl"
    hierarchy_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "domain": "grid_operations",
                        "path": ["energy_systems", "grid_operations"],
                        "members": ["grid_operations"],
                        "traits": ["coordination"],
                    }
                ),
            )
        )
    )
    with pytest.raises(ValueError, match=r"conflicts with domain identifier"):
        load_validation_records(
            Path("src/semantic_lexicon/data/cross_domain_validation.jsonl"),
            hierarchy_path=hierarchy_path,
        )


def test_domain_hierarchy_path_must_include_domain(tmp_path) -> None:
    hierarchy_path = tmp_path / "hierarchy.jsonl"
    hierarchy_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "domain": "grid_operations",
                        "path": ["energy_systems", "operations"],
                        "members": ["virtual_power_plant"],
                        "traits": ["coordination"],
                    }
                ),
            )
        )
    )
    with pytest.raises(ValueError, match=r"must end with the domain name"):
        load_validation_records(
            Path("src/semantic_lexicon/data/cross_domain_validation.jsonl"),
            hierarchy_path=hierarchy_path,
        )


def test_domain_hierarchy_member_casefold_conflict(tmp_path) -> None:
    hierarchy_path = tmp_path / "hierarchy.jsonl"
    hierarchy_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "domain": "grid_operations",
                        "path": ["energy_systems", "grid_operations"],
                        "members": ["Shared_Member"],
                        "traits": ["coordination"],
                    }
                ),
                json.dumps(
                    {
                        "domain": "demand_response",
                        "path": ["energy_systems", "demand_response"],
                        "members": ["shared_member"],
                        "traits": ["engagement"],
                    }
                ),
            )
        )
    )
    with pytest.raises(ValueError, match=r"Contradictory member 'shared_member'"):
        load_validation_records(
            Path("src/semantic_lexicon/data/cross_domain_validation.jsonl"),
            hierarchy_path=hierarchy_path,
        )


def test_domain_hierarchy_lineage_traits_propagate(tmp_path) -> None:
    hierarchy_path = tmp_path / "hierarchy.jsonl"
    hierarchy_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "domain": "science",
                        "path": ["knowledge", "science"],
                        "members": [],
                        "traits": ["domain_specific"],
                        "lineage_traits": {
                            "knowledge": ["shared_context"],
                            "knowledge>science": ["refined_focus"],
                        },
                    }
                ),
            )
        )
    )
    validation_path = tmp_path / "validation.jsonl"
    validation_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "text": "Summarise",
                        "intent": "summary",
                        "domain": "science",
                    }
                ),
            )
        )
    )
    records = load_validation_records(validation_path, hierarchy_path=hierarchy_path)
    assert records[0].traits == ("shared_context", "refined_focus", "domain_specific")


def test_domain_hierarchy_lineage_traits_prefix_must_align(tmp_path) -> None:
    hierarchy_path = tmp_path / "hierarchy.jsonl"
    hierarchy_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "domain": "science",
                        "path": ["knowledge", "science"],
                        "members": [],
                        "traits": ["domain_specific"],
                        "lineage_traits": {
                            "knowledge>mismatch": ["shared_context"],
                        },
                    }
                ),
            )
        )
    )
    with pytest.raises(ValueError, match=r"Lineage trait prefix 'knowledge>mismatch'"):
        load_validation_records(
            Path("src/semantic_lexicon/data/cross_domain_validation.jsonl"),
            hierarchy_path=hierarchy_path,
        )


def test_validation_member_identifier_cannot_be_blank(tmp_path) -> None:
    hierarchy_path = tmp_path / "hierarchy.jsonl"
    hierarchy_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "domain": "grid_operations",
                        "path": ["energy_systems", "grid_operations"],
                        "members": ["virtual_power_plant"],
                        "traits": ["coordination"],
                    }
                ),
            )
        )
    )
    validation_path = tmp_path / "validation.jsonl"
    validation_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "text": "Summarise",
                        "intent": "summary",
                        "domain": "grid_operations",
                        "domain_member": "   ",
                    }
                ),
            )
        )
    )
    with pytest.raises(ValueError, match=r"'domain_member' must be a non-empty string"):
        load_validation_records(
            validation_path,
            hierarchy_path=hierarchy_path,
        )


def test_validation_member_requires_hierarchy(tmp_path) -> None:
    validation_path = tmp_path / "validation.jsonl"
    validation_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "text": "Summarise",
                        "intent": "summary",
                        "domain": "grid_operations",
                        "domain_member": "virtual_power_plant",
                    }
                ),
            )
        )
    )
    with pytest.raises(ValueError, match="no hierarchy metadata"):
        load_validation_records(validation_path, hierarchy_path=tmp_path / "missing.jsonl")

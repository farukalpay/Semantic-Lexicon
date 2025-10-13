"""Tests for presentation planning helpers."""

from __future__ import annotations

from semantic_lexicon.presentation import build_single_adjustment_plan


def test_single_adjustment_plan_structure() -> None:
    experiment, backups = build_single_adjustment_plan()

    assert experiment.focus == "story beats"
    assert "note" in " ".join(experiment.data_to_collect).lower()
    assert "pass" in experiment.pass_fail_rule.lower()

    assert len(backups) == 5
    labels = {move.label for move in backups}
    expected = {
        "Energy checkpoints",
        "Slide trim for mixed room",
        "Q&A guardrail",
        "Warmth micro-story",
        "Lighting and breathing tweak",
    }
    assert labels == expected

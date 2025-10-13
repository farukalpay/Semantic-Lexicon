# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""End-to-end tests for the calibrated intent routing stack."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from semantic_lexicon.api import FeedbackService
from semantic_lexicon.intent import IntentClassifier, IntentExample

GROUND_TRUTH = {
    "Clarify when to use breadth-first search": "definition",
    "How should I start researching renewable energy?": "how_to",
    "Compare supervised and unsupervised learning": "comparison",
    "Offer reflective prompts for creative writing": "exploration",
}


def load_examples() -> list[IntentExample]:
    dataset = []
    path = Path("src/semantic_lexicon/data/intent.jsonl")
    with path.open("r", encoding="utf8") as handle:
        for line in handle:
            record = json.loads(line)
            dataset.append(
                IntentExample(
                    text=str(record["text"]),
                    intent=str(record["intent"]),
                    feedback=0.92,
                )
            )
    return dataset


def test_classifier_calibration_and_rewards() -> None:
    classifier = IntentClassifier()
    classifier.fit(load_examples())

    raw_ece, calibrated_ece = classifier.calibration_report
    assert calibrated_ece <= raw_ece * 0.5
    assert classifier.training_accuracy_curve
    assert classifier.training_accuracy_curve[-1] >= 0.9

    for prompt, expected in GROUND_TRUTH.items():
        predicted = classifier.predict(prompt)
        assert predicted == expected
        reward = classifier.reward(prompt, predicted, expected, feedback=0.92)
        assert reward > 0.7


def test_feedback_service_updates_weights() -> None:
    classifier = IntentClassifier()
    classifier.fit(load_examples())
    service = FeedbackService(classifier)
    before = classifier.reward_weights
    result = service.submit(
        "Compare supervised and unsupervised learning",
        "comparison",
        "comparison",
        0.96,
    )
    after = classifier.reward_weights
    assert not np.allclose(before, after)
    assert "weights" in result

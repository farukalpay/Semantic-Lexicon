# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

from __future__ import annotations

from semantic_lexicon.template_learning import (
    BalancedTutorExample,
    BalancedTutorPredictor,
)


def _examples() -> list[BalancedTutorExample]:
    return [
        BalancedTutorExample(
            prompt="How do I improve my public speaking?",
            intent="how_to",
            topics=("Public Speaking", "Practice Routine", "Feedback Loops"),
            actions=("Explore", "Practice", "Reflect"),
        ),
        BalancedTutorExample(
            prompt="Explain matrix multiplication",
            intent="definition",
            topics=("Matrix Multiplication", "Dot Products", "Linear Transformations"),
            actions=("Define", "Explore", "Compare"),
        ),
        BalancedTutorExample(
            prompt="Clarify the concept of photosynthesis",
            intent="definition",
            topics=("Photosynthesis", "Chlorophyll Function", "Energy Conversion"),
            actions=("Define", "Explore", "Connect"),
        ),
    ]


def test_predict_variables_returns_expected_tuple_for_match() -> None:
    predictor = BalancedTutorPredictor(_examples())

    variables = predictor.predict_variables("Explain matrix multiplication")

    assert variables.intent == "definition"
    assert variables.topics == (
        "Matrix Multiplication",
        "Dot Products",
        "Linear Transformations",
    )
    assert variables.actions == ("Define", "Explore", "Compare")


def test_predict_variables_is_case_insensitive_and_generalises_tokens() -> None:
    predictor = BalancedTutorPredictor(_examples())

    variables = predictor.predict_variables("How can I get better at public speaking events?")

    assert variables.intent == "how_to"
    assert variables.topics[0] == "Public Speaking"
    assert variables.actions[0] == "Explore"


def test_predict_returns_template_with_original_prompt() -> None:
    predictor = BalancedTutorPredictor(_examples())

    template = predictor.predict("Need help understanding plant energy conversion")

    assert template.prompt == "Need help understanding plant energy conversion"
    assert template.intent == "definition"
    assert "Energy Conversion" in template.topics


def test_load_default_predicts_presentation_topics() -> None:
    predictor = BalancedTutorPredictor.load_default()

    variables = predictor.predict_variables(
        "Need tips to organize my upcoming research presentation"
    )

    assert variables.intent == "how_to"
    assert "Presentation Outline" in variables.topics
    assert "Design" in variables.actions or "Practice" in variables.actions

# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

from __future__ import annotations

import pytest

from semantic_lexicon.templates import (
    BalancedTutorTemplate,
    render_balanced_tutor_response,
)


def test_render_balanced_tutor_response_matches_expected_structure() -> None:
    response = render_balanced_tutor_response(
        prompt="How do I improve my public speaking?",
        intent="how_to",
        topics=("Public Speaking", "Practice Routine", "Feedback Loops"),
        actions=("Explore", "Practice", "Reflect"),
    )

    expected = (
        "From a balanced tutor perspective, let's look at "
        '"How do I improve my public speaking?" '
        'This ties closely to the "how_to" intent I detected. '
        "Consider journaling about: Public Speaking (Explore), "
        "Practice Routine (Practice), Feedback Loops (Reflect). "
        "Try to explore Public Speaking, practice the routine, "
        "and reflect on Feedback Loops."
    )

    assert response == expected


def test_render_balanced_tutor_response_infers_missing_punctuation() -> None:
    response = render_balanced_tutor_response(
        prompt="Explain matrix multiplication",
        intent="definition",
        topics=("Matrix Multiplication", "Dot Products"),
        actions=("Define", "Explore"),
    )

    assert response.startswith(
        'From a balanced tutor perspective, let\'s look at "Explain matrix multiplication."'
    )


def test_template_dataclass_renders_using_helper() -> None:
    template = BalancedTutorTemplate(
        prompt="What is machine learning?",
        intent="definition",
        topics=(
            "Machine Learning",
            "Supervised Learning",
            "Generalization Error",
        ),
        actions=("Define", "Explore", "Compare"),
    )

    expected_suffix = (
        "Try to define Machine Learning, "
        "explore Supervised Learning, "
        "and compare Generalization Error with related ideas."
    )

    assert template.render().endswith(expected_suffix)


def test_render_balanced_tutor_response_requires_pairs() -> None:
    with pytest.raises(ValueError):
        render_balanced_tutor_response(
            prompt="",
            intent="",
            topics=(),
            actions=(),
        )

    with pytest.raises(ValueError):
        render_balanced_tutor_response(
            prompt="Topic",
            intent="general",
            topics=("Topic A",),
            actions=("Explore", "Reflect"),
        )

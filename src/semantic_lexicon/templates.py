# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""Template helpers for structured response generation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["BalancedTutorTemplate", "render_balanced_tutor_response"]


_INTRO_PREFIX = "From a balanced tutor perspective, let's look at"
_INTRO_SUFFIX = 'This ties closely to the "{intent}" intent I detected.'


_ACTION_FRAGMENT_TEMPLATES: dict[str, str] = {
    "reflect": "reflect on {topic}",
    "compare": "compare {topic} with related ideas",
    "connect": "connect {topic} to what you already know",
    "illustrate": "illustrate {topic} with a quick example",
}


@dataclass(frozen=True)
class BalancedTutorTemplate:
    """Lightweight container for templated balanced-tutor responses."""

    prompt: str
    intent: str
    topics: Sequence[str]
    actions: Sequence[str]

    def render(self) -> str:
        """Render the template into a natural-language response."""

        return render_balanced_tutor_response(
            prompt=self.prompt,
            intent=self.intent,
            topics=self.topics,
            actions=self.actions,
        )


def render_balanced_tutor_response(
    *, prompt: str, intent: str, topics: Sequence[str], actions: Sequence[str]
) -> str:
    """Render the canonical balanced-tutor journaling prompt template.

    The response is built according to the dataset pattern described in the
    documentation. The structure is::

        From a balanced tutor perspective, let's look at "<prompt>". This ties
        closely to the "<intent>" intent I detected. Consider journaling about:
        <topic1> (<action1>), <topic2> (<action2>), <topic3> (<action3>). Try to
        <action1_phrase>, <action2_phrase>, and <action3_phrase>.

    Parameters
    ----------
    prompt:
        The learner's original prompt.
    intent:
        The detected intent label.
    topics:
        Sequence of topics highlighted for journaling (1-3 entries expected).
    actions:
        Sequence of actions paired with each topic. Must match ``topics`` in
        length.
    """

    normalised_prompt = _normalise_prompt(prompt)
    normalised_intent = intent or "general"
    topic_pairs = _validate_topics_and_actions(topics, actions)

    intro = _format_intro(normalised_prompt, normalised_intent)
    journaling = _format_journaling(topic_pairs)
    try_sentence = _format_try_sentence(topic_pairs)

    return f"{intro} {journaling} {try_sentence}".strip()


def _normalise_prompt(prompt: str) -> str:
    text = prompt.strip()
    if not text:
        text = "this topic"
    if text[-1] not in ".!?":
        text = f"{text}."
    return text


def _validate_topics_and_actions(
    topics: Sequence[str], actions: Sequence[str]
) -> list[tuple[str, str]]:
    if len(topics) != len(actions):
        raise ValueError("topics and actions must be the same length")
    if not topics:
        raise ValueError("at least one topic/action pair is required")
    return [(topic, action) for topic, action in zip(topics, actions)]


def _format_intro(prompt: str, intent: str) -> str:
    return f'{_INTRO_PREFIX} "{prompt}" {_INTRO_SUFFIX.format(intent=intent)}'


def _format_journaling(pairs: Sequence[tuple[str, str]]) -> str:
    joined = ", ".join(f"{topic} ({action})" for topic, action in pairs)
    return f"Consider journaling about: {joined}."


def _format_try_sentence(pairs: Sequence[tuple[str, str]]) -> str:
    fragments = [_format_action_fragment(topic, action) for topic, action in pairs]
    if len(fragments) == 1:
        tail = fragments[0]
    elif len(fragments) == 2:
        tail = " and ".join(fragments)
    else:
        tail = f"{', '.join(fragments[:-1])}, and {fragments[-1]}"
    return f"Try to {tail}."


def _format_action_fragment(topic: str, action: str) -> str:
    key = action.strip().lower()
    template = _ACTION_FRAGMENT_TEMPLATES.get(key)
    if template is not None:
        return template.format(topic=topic)
    action_lower = key or action.lower()
    topic_words = topic.split()
    if topic_words and topic_words[0].lower() == action_lower:
        trimmed = " ".join(topic_words[1:]).strip()
        if trimmed:
            normalised = trimmed[0].lower() + trimmed[1:]
            return f"{action_lower} the {normalised}"
    return f"{action_lower} {topic}"

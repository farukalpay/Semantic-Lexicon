"""Planning utilities for high-stakes presentation tune-ups."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "ExperimentPlan",
    "BackupMove",
    "build_single_adjustment_plan",
]


@dataclass(frozen=True)
class ExperimentPlan:
    """Primary tweak for the next run plus an executable experiment."""

    focus: str
    rationale: str
    steps: Sequence[str]
    data_to_collect: Sequence[str]
    pass_fail_rule: str


@dataclass(frozen=True)
class BackupMove:
    """Intent-hidden contingency move for fast rehearsal pivots."""

    label: str
    description: str
    protocol: Sequence[str]


def build_single_adjustment_plan() -> tuple[ExperimentPlan, list[BackupMove]]:
    """Return the recommended one-change plan plus backup drills."""

    experiment = ExperimentPlan(
        focus="story beats",
        rationale=(
            "Note-taking tailed off after minute 7, which usually means the mid-"
            "talk beat stopped delivering new stakes. Re-mapping the beats keeps"
            "one changeable lever while pacing and visuals stay fixed."
        ),
        steps=(
            "Sketch a 12-minute beat grid: 0-2 hook, 2-6 tension with evidence,"
            " 6-9 solution reveal, 9-11 proof, 11-12 closing ask.",
            "Record a timed rehearsal following the new beats. After each 2-minute"
            " block, pause 30 seconds to mark a headline the audience should write"
            " down next and note your perceived energy (1-5).",
            "Review the five headlines: ensure at least one fresh, specific takeaway"
            " lands after minute 7 and the close still points to the same ask.",
        ),
        data_to_collect=(
            "Five 2-minute timestamps with the headline you expect them to note",
            "Self-rated energy (1-5) for each block",
            "Count of distinct note-worthy headlines appearing after minute 7",
        ),
        pass_fail_rule=(
            "Pass if two or more distinct note-worthy headlines appear after minute"
            " 7 and energy stays at 4+ through the 9-11 block; otherwise rerun with"
            " sharper stakes or a trimmed example."
        ),
    )

    backups = [
        BackupMove(
            label="Energy checkpoints",
            description=(
                "Use tonight's 40 minutes to add two checkpoints that predict a strong closer."
            ),
            protocol=(
                "Run two 15-minute partial rehearsals: stop after slide 5 and"
                " again at the penultimate slide.",
                "At each stop, rate breath control, vocal brightness, and whether"
                " you can restate the closing promise without notes (1/0).",
                "If the second checkpoint scores lower than the first on two of"
                " three metrics, script a micro-reset before the finale.",
            ),
        ),
        BackupMove(
            label="Slide trim for mixed room",
            description=(
                "Remove the data-dense architecture slide to protect narrative"
                " flow across engineering and exec listeners."
            ),
            protocol=(
                "Delete the architecture slide; open with problem framing and"
                " keep the ROI slide as proof.",
                "Replace the deleted slide with a single line: 'Hand-off kit maps"
                " architecture in appendix if you want the wiring.'",
            ),
        ),
        BackupMove(
            label="Q&A guardrail",
            description="Prevent rambling answers after interruptions.",
            protocol=(
                "Practice with a 90-second timer: 0-30s repeat the question,"
                " 30-75s give the answer, 75-90s bridge back to your close.",
                "Use the stop phrase 'I'll park the rest for follow-up' when the"
                " timer hits 85 seconds.",
            ),
        ),
        BackupMove(
            label="Warmth micro-story",
            description="Quick story beat to reintroduce warmth without slowing down.",
            protocol=(
                "Template: 'Two weeks ago [role] said “__”. I borrowed their move;"
                " here’s the 30-second result.'",
            ),
        ),
        BackupMove(
            label="Lighting and breathing tweak",
            description="Keep filler words in check under dim lighting.",
            protocol=(
                "Raise a desk lamp to 120 lux aimed at eye level; before speaking"
                " run three 4-7-8 breaths facing the light.",
                "Record a one-minute warm-up both before and after; tally filler"
                " words to confirm the drop.",
            ),
        ),
    ]

    return experiment, backups

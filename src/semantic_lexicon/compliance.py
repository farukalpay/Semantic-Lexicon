"""Utilities for generating structured compliance reports.

The project often needs to persist the results of behavioral compliance
evaluations.  The helper functions in this module take care of rendering the
summary and individual test cases in both JSON and Markdown formats while
guarding against the formatting pitfalls that can easily creep into string
interpolation code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence
import json


def _format_case_line(case: Mapping[str, object]) -> str:
    """Return a single Markdown list item describing a compliance case.

    The previous ad-hoc script attempted to interpolate dictionary keys without
    quoting them, which results in a ``SyntaxError`` at runtime.  This helper
    provides a well-tested and reusable implementation.
    """

    label = str(case.get("label", "(unknown)"))
    status = "PASS" if case.get("pass") else "FAIL"
    notes = case.get("notes", {})
    notes_blob = json.dumps(notes, sort_keys=True)
    return f"- **{label}** — {status} — {notes_blob}"


def render_markdown(summary: Mapping[str, object], cases: Sequence[Mapping[str, object]]) -> str:
    """Render a compliance report to Markdown."""

    lines = ["# Semantic-Lexicon Compliance Report", "", "## Summary"]
    for key, value in summary.items():
        lines.append(f"- **{key}**: {value}")

    lines.append("")
    lines.append("## Cases")
    for case in cases:
        lines.append(_format_case_line(case))

    return "\n".join(lines)


def write_reports(
    summary: Mapping[str, object],
    cases: Sequence[Mapping[str, object]],
    json_path: Path | str,
    markdown_path: Path | str,
) -> None:
    """Persist the compliance report to JSON and Markdown files."""

    data = {"summary": dict(summary), "cases": list(cases)}
    Path(json_path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf8")
    Path(markdown_path).write_text(render_markdown(summary, cases) + "\n", encoding="utf8")


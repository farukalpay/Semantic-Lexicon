"""Utilities to compile compliance reports.

The real project contains a number of small evaluation harnesses that emit
``summary`` dictionaries alongside per-case diagnostics.  A helper module keeps
the Markdown and JSON rendering logic in one place so callers only need to feed
Python data structures.  The previous ad-hoc script used ``f"{c[label]}"``
syntax, which is invalid because dictionary keys must be referenced with
quotes.  This module centralises the implementation and protects against such
syntax slips.
"""

from __future__ import annotations

import json
import sys
from collections import abc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_DATACLASS_EXTRA_KWARGS: dict[str, Any] = {}
if sys.version_info >= (3, 10):  # pragma: no cover - depends on runtime version
    _DATACLASS_EXTRA_KWARGS["slots"] = True


@dataclass(frozen=True, **_DATACLASS_EXTRA_KWARGS)
class CaseRecord:
    """Individual compliance evaluation result."""

    label: str
    passed: bool
    notes: abc.Mapping[str, Any]

    def status_text(self) -> str:
        """Return the Markdown-friendly status string."""

        return "PASS" if self.passed else "FAIL"


@dataclass(frozen=True, **_DATACLASS_EXTRA_KWARGS)
class ComplianceSummary:
    """Aggregate metrics for a compliance report."""

    total: int
    passed: int
    failed: int
    pass_rate: float


def _summary_lines(summary: ComplianceSummary) -> list[str]:
    """Create Markdown bullet points for the summary section."""

    return [f"- **{field}**: {value}" for field, value in asdict(summary).items()]


def _case_lines(cases: abc.Sequence[CaseRecord]) -> list[str]:
    """Create Markdown bullet points for each evaluation case."""

    if not isinstance(cases, abc.Sequence):
        raise TypeError("cases must be a sequence of CaseRecord instances")

    lines: list[str] = []
    for case in cases:
        if not isinstance(case, CaseRecord):
            raise TypeError("each case must be a CaseRecord instance")
        if not isinstance(case.notes, abc.Mapping):
            raise TypeError("case notes must be a mapping")
        lines.append(f"- **{case.label}** — {case.status_text()} — {json.dumps(case.notes)}")
    return lines


def build_markdown(summary: ComplianceSummary, cases: abc.Sequence[CaseRecord]) -> str:
    """Compose the Markdown document for a compliance report."""

    lines = ["# Semantic-Lexicon Compliance Report", "", "## Summary"]
    lines.extend(_summary_lines(summary))
    lines.append("")
    lines.append("## Cases")
    lines.extend(_case_lines(cases))
    return "\n".join(lines)


def build_json(summary: ComplianceSummary, cases: abc.Sequence[CaseRecord]) -> str:
    """Compose the JSON document for a compliance report."""

    payload = {
        "summary": asdict(summary),
        "cases": [asdict(case) for case in cases],
    }
    return json.dumps(payload, indent=2)


def write_reports(
    summary: ComplianceSummary,
    cases: abc.Sequence[CaseRecord],
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    """Persist the compliance reports to disk."""

    json_path = Path(json_path)
    markdown_path = Path(markdown_path)

    json_payload = build_json(summary, cases)
    markdown_payload = build_markdown(summary, cases)

    json_path.write_text(json_payload, encoding="utf8")
    markdown_path.write_text(markdown_payload, encoding="utf8")

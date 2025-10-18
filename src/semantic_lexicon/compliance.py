"""Utilities to compile compliance reports.

The project ships with evaluation harnesses that emit ``summary`` dictionaries
alongside per-case diagnostics. Centralising the Markdown and JSON rendering
logic avoids the brittle inline snippets such as ``f"{c[label]}"`` which raise
syntax errors because dictionary keys need to be quoted.
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


def _parse_summary(payload: abc.Mapping[str, Any]) -> ComplianceSummary:
    """Coerce a mapping into :class:`ComplianceSummary`."""

    required_fields = {"total", "passed", "failed", "pass_rate"}
    missing = required_fields.difference(payload)
    if missing:
        msg = f"summary is missing fields: {sorted(missing)}"
        raise KeyError(msg)

    total = int(payload["total"])
    passed = int(payload["passed"])
    failed = int(payload["failed"])
    pass_rate = float(payload["pass_rate"])
    return ComplianceSummary(
        total=total,
        passed=passed,
        failed=failed,
        pass_rate=pass_rate,
    )


def _parse_case(payload: abc.Mapping[str, Any]) -> CaseRecord:
    """Coerce a mapping into :class:`CaseRecord`."""

    required_fields = {"label", "passed"}
    missing = required_fields.difference(payload)
    if missing:
        msg = f"case entry is missing fields: {sorted(missing)}"
        raise KeyError(msg)

    label = str(payload["label"])
    passed = bool(payload["passed"])
    notes = payload.get("notes", {})
    if notes is None:
        notes = {}
    if not isinstance(notes, abc.Mapping):
        raise TypeError("case notes must be a mapping")
    return CaseRecord(label=label, passed=passed, notes=dict(notes))


def _ensure_iterable(value: Any) -> abc.Iterable[Any]:
    if isinstance(value, (str, bytes)):
        raise TypeError("cases must be a sequence of mappings")
    if not isinstance(value, abc.Iterable):
        raise TypeError("cases must be a sequence of mappings")
    return value


def load_report_payload(path: Path) -> tuple[ComplianceSummary, list[CaseRecord]]:
    """Load summary and cases from a JSON payload."""

    data = json.loads(Path(path).read_text(encoding="utf8"))
    if not isinstance(data, abc.Mapping):
        raise TypeError("payload must be a JSON object with 'summary' and 'cases'")

    summary_payload = data.get("summary")
    cases_payload = data.get("cases")
    if not isinstance(summary_payload, abc.Mapping):
        raise TypeError("summary must be a mapping")
    if cases_payload is None:
        cases_payload = []
    iterable_cases = _ensure_iterable(cases_payload)

    summary = _parse_summary(summary_payload)
    cases = []
    for index, entry in enumerate(iterable_cases):
        if not isinstance(entry, abc.Mapping):
            msg = f"case at index {index} must be a mapping"
            raise TypeError(msg)
        cases.append(_parse_case(entry))
    return summary, cases


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

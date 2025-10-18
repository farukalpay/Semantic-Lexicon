from __future__ import annotations

import json
from pathlib import Path

from semantic_lexicon.compliance import (
    CaseRecord,
    ComplianceSummary,
    build_json,
    build_markdown,
    load_report_payload,
    write_reports,
)


def _sample_data() -> tuple[ComplianceSummary, list[CaseRecord]]:
    summary = ComplianceSummary(total=3, passed=2, failed=1, pass_rate=66.7)
    cases = [
        CaseRecord(label="alpha", passed=True, notes={"detail": "ok"}),
        CaseRecord(label="beta", passed=False, notes={"error": "mismatch"}),
        CaseRecord(label="gamma", passed=True, notes={"warning": False}),
    ]
    return summary, cases


def test_build_markdown_contains_expected_sections() -> None:
    summary, cases = _sample_data()
    markdown = build_markdown(summary, cases)

    assert "# Semantic-Lexicon Compliance Report" in markdown
    assert markdown.count("## Summary") == 1
    assert markdown.count("## Cases") == 1
    assert "- **alpha** — PASS" in markdown
    assert "- **beta** — FAIL" in markdown


def test_build_json_serialises_payload() -> None:
    summary, cases = _sample_data()
    payload = json.loads(build_json(summary, cases))

    assert payload["summary"]["passed"] == 2
    assert payload["cases"][1]["label"] == "beta"
    assert payload["cases"][1]["passed"] is False


def test_write_reports(tmp_path: Path) -> None:
    summary, cases = _sample_data()
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"

    write_reports(summary, cases, json_path=json_path, markdown_path=markdown_path)

    assert json_path.exists()
    assert markdown_path.exists()
    assert "PASS" in markdown_path.read_text(encoding="utf8")


def test_load_report_payload(tmp_path: Path) -> None:
    payload = {
        "summary": {"total": 2, "passed": 1, "failed": 1, "pass_rate": 50.0},
        "cases": [
            {"label": "alpha", "passed": True, "notes": {"detail": "ok"}},
            {"label": "beta", "passed": False, "notes": {"error": "mismatch"}},
        ],
    }
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(payload), encoding="utf8")

    summary, cases = load_report_payload(path)

    assert summary.passed == 1
    assert len(cases) == 2
    assert cases[1].label == "beta"
    assert cases[1].notes["error"] == "mismatch"

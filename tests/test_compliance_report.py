from __future__ import annotations

import json
from pathlib import Path

from semantic_lexicon.compliance import render_markdown, write_reports


def test_render_markdown_produces_expected_layout(tmp_path: Path) -> None:
    summary = {"total": 10, "passed": 9, "failed": 1, "pass_rate": 90.0}
    cases = [
        {"label": "json_exact_only", "pass": True, "notes": {"output": "{\"ok\":true}"}},
        {"label": "injection_resistance", "pass": False, "notes": {"output": "OK"}},
    ]

    markdown = render_markdown(summary, cases)
    expected_lines = [
        "# Semantic-Lexicon Compliance Report",
        "",
        "## Summary",
        "- **total**: 10",
        "- **passed**: 9",
        "- **failed**: 1",
        "- **pass_rate**: 90.0",
        "",
        "## Cases",
        "- **json_exact_only** — PASS — {\"output\": \"{\\\"ok\\\":true}\"}",
        "- **injection_resistance** — FAIL — {\"output\": \"OK\"}",
    ]
    assert markdown.splitlines() == expected_lines

    json_path = tmp_path / "compliance.json"
    md_path = tmp_path / "compliance.md"
    write_reports(summary, cases, json_path, md_path)

    data = json.loads(json_path.read_text(encoding="utf8"))
    assert data == {"summary": summary, "cases": cases}
    assert md_path.read_text(encoding="utf8").rstrip("\n") == markdown


# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from semantic_lexicon.cli import app

runner = CliRunner()


def test_cli_train_and_diagnostics(tmp_path: Path) -> None:
    workspace = tmp_path / "artifacts"
    workspace.mkdir()
    # copy sample data
    intent_src = Path("src/semantic_lexicon/data/intent.jsonl")
    knowledge_src = Path("src/semantic_lexicon/data/knowledge.jsonl")
    intent_dst = workspace / "intent.jsonl"
    knowledge_dst = workspace / "knowledge.jsonl"
    intent_dst.write_text(intent_src.read_text())
    knowledge_dst.write_text(knowledge_src.read_text())

    train_result = runner.invoke(app, ["train", "--workspace", str(workspace)])
    assert train_result.exit_code == 0
    diag_result = runner.invoke(app, ["diagnostics", "--workspace", str(workspace)])
    assert diag_result.exit_code == 0
    assert "embedding_stats" in diag_result.stdout
    knowledge_result = runner.invoke(
        app,
        [
            "knowledge",
            "Explain AI",
            "--workspace",
            str(workspace),
        ],
    )
    assert knowledge_result.exit_code == 0
    assert '"concepts"' in knowledge_result.stdout


def test_cli_prepare_supports_short_option(tmp_path: Path) -> None:
    workspace = tmp_path / "artifacts"
    workspace.mkdir()
    intent_src = Path("src/semantic_lexicon/data/intent.jsonl")
    knowledge_src = Path("src/semantic_lexicon/data/knowledge.jsonl")

    result = runner.invoke(
        app,
        [
            "prepare",
            "--intent",
            str(intent_src),
            "--knowledge",
            str(knowledge_src),
            "--workspace",
            str(workspace),
        ],
    )

    assert result.exit_code == 0
    assert (workspace / "intent.jsonl").exists()
    assert (workspace / "knowledge.jsonl").exists()


def test_cli_compliance_report(tmp_path: Path) -> None:
    payload = {
        "summary": {"total": 2, "passed": 2, "failed": 0, "pass_rate": 100.0},
        "cases": [
            {"label": "alpha", "passed": True, "notes": {"detail": "ok"}},
            {"label": "beta", "passed": True, "notes": {"detail": "ok"}},
        ],
    }
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(payload), encoding="utf8")
    json_output = tmp_path / "report.json"
    markdown_output = tmp_path / "report.md"

    result = runner.invoke(
        app,
        [
            "compliance-report",
            str(payload_path),
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ],
    )

    assert result.exit_code == 0
    assert json_output.exists()
    assert markdown_output.exists()
    markdown_text = markdown_output.read_text(encoding="utf8")
    assert "- **alpha**" in markdown_text

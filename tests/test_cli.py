# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

from __future__ import annotations

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
    assert "\"concepts\"" in knowledge_result.stdout

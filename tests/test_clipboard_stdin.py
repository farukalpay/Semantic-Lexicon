# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from semantic_lexicon import cli as cli_module

runner = CliRunner()


@pytest.fixture
def dummy_generation(monkeypatch):
    """Patch model loading to capture the prompt."""

    captured: dict[str, str] = {}

    class DummyModel:
        def generate(self, prompt: str, persona: str | None = None) -> SimpleNamespace:
            captured["prompt"] = prompt
            captured["persona"] = persona or ""
            return SimpleNamespace(response="stub response")

    def fake_load_model(config_path):
        return DummyModel(), SimpleNamespace()

    monkeypatch.setattr(cli_module, "_load_model", fake_load_model)
    return captured


def test_generate_reads_stdin(dummy_generation, tmp_path: Path) -> None:
    workspace = tmp_path / "artifacts"
    workspace.mkdir()

    result = runner.invoke(
        cli_module.app,
        ["generate", "-", "--workspace", str(workspace)],
        input="What is a transformer?",
    )

    assert result.exit_code == 0
    assert dummy_generation["prompt"] == "what is a transformer?"


def test_clipboard_empty(monkeypatch, tmp_path: Path) -> None:
    workspace = tmp_path / "artifacts"
    workspace.mkdir()

    monkeypatch.setattr(cli_module, "get_clipboard_text", lambda: "")

    result = runner.invoke(
        cli_module.app,
        ["clipboard", "--workspace", str(workspace)],
    )

    assert result.exit_code != 0
    assert "Clipboard is empty." in result.output


def test_clipboard_error(monkeypatch, tmp_path: Path) -> None:
    workspace = tmp_path / "artifacts"
    workspace.mkdir()

    def raise_error():
        raise cli_module.ClipboardError("Clipboard is empty.")

    monkeypatch.setattr(cli_module, "get_clipboard_text", raise_error)

    result = runner.invoke(
        cli_module.app,
        ["clipboard", "--workspace", str(workspace)],
    )

    assert result.exit_code != 0
    assert "Clipboard is empty." in result.output

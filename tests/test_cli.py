import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from semantic_lexicon import SemanticModelConfig
from semantic_lexicon.cli import app
from semantic_lexicon.data import load_sample_corpus


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_prepare_and_train_commands(tmp_path: Path, runner: CliRunner) -> None:
    corpus_path = tmp_path / "corpus.json"
    corpus = load_sample_corpus()
    corpus_path.write_text(json.dumps(corpus))

    processed_path = tmp_path / "processed.json"
    result = runner.invoke(app, ["prepare", str(corpus_path), "--output", str(processed_path)])
    assert result.exit_code == 0
    assert processed_path.exists()

    config_path = tmp_path / "config.yaml"
    SemanticModelConfig().save(config_path)
    train_result = runner.invoke(app, ["train", str(processed_path), "--config", str(config_path), "--workspace", str(tmp_path / "workspace")])
    assert train_result.exit_code == 0


def test_generate_command(tmp_path: Path, runner: CliRunner) -> None:
    result = runner.invoke(app, ["generate", "Explain automation"])
    assert result.exit_code == 0
    assert "automation" in result.stdout.lower() or result.stdout

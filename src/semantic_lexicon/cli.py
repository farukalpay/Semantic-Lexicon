"""Command line interface for Semantic Lexicon."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from .config import SemanticModelConfig, load_config
from .logging import configure_logging
from .model import NeuralSemanticModel
from .training import Trainer, TrainerConfig
from .utils import normalise_text, read_jsonl

LOGGER = configure_logging(logger_name=__name__)

DEFAULT_WORKSPACE = Path("artifacts")
INTENT_PATH_OPTION = typer.Option(
    ..., help="Path to raw intent dataset (JSON or JSONL)."
)
KNOWLEDGE_PATH_OPTION = typer.Option(
    ..., help="Path to raw knowledge dataset."
)
WORKSPACE_OPTION = typer.Option(
    DEFAULT_WORKSPACE,
    help="Output directory for processed datasets.",
)
CONFIG_OPTION = typer.Option(
    None,
    help="Path to semantic model configuration.",
)
TRAIN_WORKSPACE_OPTION = typer.Option(
    DEFAULT_WORKSPACE,
    help="Workspace containing processed datasets.",
)
DIAG_WORKSPACE_OPTION = typer.Option(
    DEFAULT_WORKSPACE,
    help="Workspace containing trained artifacts.",
)
DIAG_OUTPUT_OPTION = typer.Option(
    None,
    help="Optional path to write diagnostics report.",
)
PERSONA_OPTION = typer.Option(
    None,
    help="Persona to condition generation.",
)
GENERATE_CONFIG_OPTION = typer.Option(
    None,
    help="Optional configuration file.",
)
GENERATE_WORKSPACE_OPTION = typer.Option(
    DEFAULT_WORKSPACE,
    help="Directory containing trained artifacts.",
)

app = typer.Typer(
    help="Automate training, diagnostics, and generation for the Semantic Lexicon model."
)


def _load_records(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() in {".jsonl", ".jsonl.gz"}:
        return list(read_jsonl(path))
    with path.open("r", encoding="utf8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        records = data.get("records", [])
        if isinstance(records, list):
            return records
        msg = "Expected 'records' list in JSON payload"
        raise TypeError(msg)
    if isinstance(data, list):
        return data
    msg = "Unsupported corpus format; expected list or mapping with 'records'"
    raise TypeError(msg)


def _load_model(config_path: Path | None) -> tuple[NeuralSemanticModel, SemanticModelConfig]:
    config = load_config(config_path)
    model = NeuralSemanticModel(config=config)
    return model, config


@app.command()
def prepare(
    intent_path: Path = INTENT_PATH_OPTION,
    knowledge_path: Path = KNOWLEDGE_PATH_OPTION,
    workspace: Path = WORKSPACE_OPTION,
) -> None:
    """Normalise raw datasets and write JSONL files suitable for training."""

    LOGGER.info("Preparing corpus")
    intents = _load_records(intent_path)
    knowledge = _load_records(knowledge_path)
    trainer_config = TrainerConfig(workspace=workspace)
    model, _ = _load_model(None)
    trainer = Trainer(model, trainer_config)
    trainer.prepare_corpus(intents, knowledge)


@app.command()
def train(
    config_path: Path | None = CONFIG_OPTION,
    workspace: Path = TRAIN_WORKSPACE_OPTION,
) -> None:
    """Train the intent classifier and knowledge network."""

    model, config = _load_model(config_path)
    trainer_config = TrainerConfig(
        workspace=workspace,
        intent_dataset=workspace / "intent.jsonl",
        knowledge_dataset=workspace / "knowledge.jsonl",
    )
    trainer = Trainer(model, trainer_config)
    trainer.train()
    trainer.model.save(workspace)


@app.command()
def diagnostics(
    config_path: Path | None = CONFIG_OPTION,
    workspace: Path = DIAG_WORKSPACE_OPTION,
    output: Path | None = DIAG_OUTPUT_OPTION,
) -> None:
    """Run diagnostics and optionally export to JSON."""

    model, _ = _load_model(config_path)
    artifacts_dir = Path(workspace)
    if (artifacts_dir / "embeddings.json").exists():
        model = NeuralSemanticModel.load(artifacts_dir, config=model.config)
    trainer = Trainer(model, TrainerConfig(workspace=workspace))
    result = trainer.run_diagnostics()
    typer.echo(json.dumps(result.to_dict(), indent=2))
    if output is not None:
        result.to_json(output)
        LOGGER.info("Wrote diagnostics to %s", output)


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Prompt to respond to."),
    persona: str | None = PERSONA_OPTION,
    config_path: Path | None = GENERATE_CONFIG_OPTION,
    workspace: Path = GENERATE_WORKSPACE_OPTION,
) -> None:
    """Generate a persona-conditioned response."""

    model, _ = _load_model(config_path)
    artifacts_dir = Path(workspace)
    if (artifacts_dir / "embeddings.json").exists():
        model = NeuralSemanticModel.load(artifacts_dir, config=model.config)
    result = model.generate(normalise_text(prompt), persona=persona)
    typer.echo(result.response)


if __name__ == "__main__":
    app()

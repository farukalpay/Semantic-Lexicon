"""Command line interface for the Semantic Lexicon."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .config import SemanticModelConfig, save_default_config
from .diagnostics import run_all_diagnostics
from .logging import configure_logging, get_logger
from .model import ModelArtifacts, NeuralSemanticModel
from .training import CorpusProcessor, Trainer
from .utils import load_yaml_or_json, save_json

app = typer.Typer(add_completion=False, help="Semantic Lexicon automation CLI")
LOGGER = get_logger(__name__)


def _load_config(path: Optional[Path]) -> SemanticModelConfig:
    if path is None:
        return SemanticModelConfig()
    return SemanticModelConfig.load(path)


@app.command()
def init_config(path: Path = typer.Argument(Path("semantic_lexicon.yaml"), help="Where to write the default config.")) -> None:
    """Write the default configuration file."""
    save_default_config(path)
    typer.echo(f"Wrote configuration to {path}")


@app.command()
def prepare(
    corpus: Path = typer.Argument(..., exists=True, readable=True, help="Path to JSON/JSONL/YAML corpus."),
    output: Path = typer.Option(Path("processed_corpus.json"), help="Path to write the processed corpus."),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to configuration file."),
) -> None:
    """Prepare a raw corpus into a processed representation."""
    configure_logging()
    _ = _load_config(config_path)
    processor = CorpusProcessor()
    data = load_yaml_or_json(corpus) if corpus.suffix.lower() in {".json", ".yaml", ".yml"} else None
    if data is None:
        raise typer.BadParameter("Unsupported corpus format; provide JSON or YAML")
    processed = processor.prepare_corpus(data if isinstance(data, list) else data["records"])
    save_json(output, {"entries": processed, "vocabulary": processor.vocabulary.most_common(1000)})
    typer.echo(f"Processed {len(processed)} entries → {output}")


@app.command()
def train(
    processed_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to processed corpus JSON."),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Model configuration."),
    workspace: Optional[Path] = typer.Option(None, help="Override workspace directory."),
) -> None:
    """Train the semantic model from processed corpus data."""
    configure_logging()
    config = _load_config(config_path)
    if workspace is not None:
        config.workspace = workspace
    payload = load_yaml_or_json(processed_path)
    entries = payload.get("entries", payload)
    model = NeuralSemanticModel(config=config)
    trainer = Trainer(model, config=config.trainer)
    trainer.train(entries)
    artifacts = model.save()
    trainer.save_training_report(config.workspace / "training_report.json")
    typer.echo(f"Training complete. Artifacts stored in {config.workspace}")
    typer.echo(json.dumps(artifacts.__dict__, default=str, indent=2))


@app.command()
def diagnostics(
    workspace: Optional[Path] = typer.Option(None, help="Load model artefacts from workspace before running diagnostics."),
    config_path: Optional[Path] = typer.Option(None, help="Configuration file."),
    output: Optional[Path] = typer.Option(None, help="Optional path to write diagnostics JSON."),
) -> None:
    """Run the diagnostics suite."""
    configure_logging()
    config = _load_config(config_path)
    model = NeuralSemanticModel(config=config)
    if workspace is not None:
        artifacts = ModelArtifacts(
            embeddings_path=workspace / "embeddings.npz",
            intent_path=workspace / "intent.npz",
            knowledge_path=workspace / "knowledge.json",
        )
        model = NeuralSemanticModel.load(artifacts=artifacts, config=config)
    result = run_all_diagnostics(model=model)
    if output:
        save_json(output, result.to_dict())
        typer.echo(f"Diagnostics written to {output}")


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Prompt or query to generate a response for."),
    persona: Optional[str] = typer.Option(None, help="Persona label."),
    config_path: Optional[Path] = typer.Option(None, help="Configuration file."),
) -> None:
    """Generate a response using the semantic model."""
    config = _load_config(config_path)
    model = NeuralSemanticModel(config=config)
    response = model.generate_response(prompt, persona=persona)
    typer.echo(response)


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()

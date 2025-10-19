# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""Command line interface for Semantic Lexicon."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from collections.abc import Iterable

import typer  # type: ignore[import-not-found]

from .compliance import load_report_payload, write_reports
from .config import SemanticModelConfig, load_config
from .logging import configure_logging
from .model import NeuralSemanticModel
from .training import Trainer, TrainerConfig
from .utils import normalise_text, read_jsonl
from .utils.clipboard import ClipboardError, get_clipboard_text

LOGGER = configure_logging(logger_name=__name__)

DEFAULT_WORKSPACE = Path("artifacts")
INTENT_PATH_OPTION = typer.Option(
    ...,
    "--intent",
    "--intent-path",
    help="Path to raw intent dataset (JSON or JSONL).",
)
KNOWLEDGE_PATH_OPTION = typer.Option(
    ...,
    "--knowledge",
    "--knowledge-path",
    help="Path to raw knowledge dataset.",
)
WORKSPACE_OPTION = typer.Option(
    DEFAULT_WORKSPACE,
    help="Output directory for processed datasets.",
)
CONFIG_OPTION = typer.Option(
    None,
    "--config",
    "--config-path",
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
CLIPBOARD_PERSONA_OPTION = typer.Option(
    "generic",
    help="Persona to condition generation.",
)
GENERATE_CONFIG_OPTION = typer.Option(
    None,
    "--config",
    "--config-path",
    help="Optional configuration file.",
)
GENERATE_WORKSPACE_OPTION = typer.Option(
    DEFAULT_WORKSPACE,
    help="Directory containing trained artifacts.",
)
CLIPBOARD_WORKSPACE_OPTION = typer.Option(
    ...,
    "--workspace",
    exists=True,
    file_okay=False,
    dir_okay=True,
    resolve_path=True,
    help="Directory containing trained artifacts.",
)
BULLET_COUNT_OPTION = typer.Option(
    2,
    "--bullets",
    min=1,
    max=5,
    help="Number of actionable bullet points to return (1-5).",
)
COMPLIANCE_JSON_OUTPUT_OPTION = typer.Option(
    DEFAULT_WORKSPACE / "compliance.json",
    "--json-output",
    help="Destination path for the compliance JSON report.",
)
COMPLIANCE_MARKDOWN_OUTPUT_OPTION = typer.Option(
    DEFAULT_WORKSPACE / "compliance.md",
    "--markdown-output",
    help="Destination path for the compliance Markdown report.",
)
COMPLIANCE_PAYLOAD_ARGUMENT = typer.Argument(
    ..., help="JSON file containing 'summary' and 'cases' payloads."
)

app = typer.Typer(
    help="Automate training, diagnostics, and generation for the Semantic Lexicon model."
)


_CONCEPT_HINTS_EXACT = {
    "weight decay": "Use weight decay (AdamW)—start near 0.01 and tune.",
    "l2": "Use weight decay (AdamW)—start near 0.01 and tune.",
    "regularization": "Use weight decay (AdamW)—start near 0.01 and tune.",
    "dropout": "Add dropout (≈0.2–0.5) and tune on validation.",
    "early stopping": "Enable early stopping on val loss (patience 5–10).",
    "small learning rate": "Lower LR; use warmup + cosine/step decay.",
    "learning rate": "Lower LR; use warmup + cosine/step decay.",
    "batch size": "Tune batch size: larger = stability, smaller = regularization.",
}

_CONCEPT_HINTS_PREFIX = {
    "data augmentation": "Apply augmentation (flip/crop/jitter) to expand data.",
    "fine tuning": "Freeze lower layers first; unfreeze gradually while tracking val.",
}

_QUESTION_FALLBACKS = [
    (
        ("overfitting",),
        [
            "Use weight decay (AdamW)—start near 0.01 and tune.",
            "Add augmentation and early stopping on validation.",
        ],
    ),
    (
        ("transformer", "fine-tune", "finetune", "fine tune"),
        [
            "Lower LR; freeze lower layers then unfreeze progressively.",
            "Use AdamW with weight decay; add LR warmup + decay.",
        ],
    ),
]

_DEFAULT_FALLBACKS = [
    "Define metric + val split; optimize only what you measure.",
    "Start with a small baseline; change one thing at a time.",
]


def _normalise_phrase(text: str) -> str:
    cleaned = text.strip().lower().replace("_", " ")
    return " ".join(cleaned.replace("-", " ").split())


def _push_unique(items: list[str], value: str) -> None:
    key = value.lower()
    if key not in {existing.lower() for existing in items}:
        items.append(value)


def _iter_candidate_phrases(response: str) -> Iterable[str]:
    for raw_line in response.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("knowledge focus:"):
            payload = line.split(":", 1)[1].strip().rstrip(".,")
            if payload:
                yield payload
            continue
        if "related concepts worth exploring:" in line.lower():
            _, payload = line.split(":", 1)
            for chunk in payload.strip().rstrip(".").split(","):
                chunk = chunk.strip()
                if chunk:
                    yield chunk
            continue
        if line.startswith("- "):
            yield line[2:].strip()


def _concept_to_hint(concept: str) -> Optional[str]:
    key = _normalise_phrase(concept)
    if not key:
        return None
    if key in _CONCEPT_HINTS_EXACT:
        return _CONCEPT_HINTS_EXACT[key]
    for prefix, hint in _CONCEPT_HINTS_PREFIX.items():
        if key.startswith(prefix):
            return hint
    return None


def _question_fallbacks(question: str) -> Iterable[str]:
    lower_question = question.lower()
    for keywords, hints in _QUESTION_FALLBACKS:
        if any(keyword in lower_question for keyword in keywords):
            yield from hints
            return
    yield from _DEFAULT_FALLBACKS


def _tight_bullet_points(
    prompt: str,
    response: str,
    knowledge_concepts: Iterable[str],
    limit: int,
) -> list[str]:
    bullets: list[str] = []
    for concept in knowledge_concepts:
        hint = _concept_to_hint(concept)
        if hint:
            _push_unique(bullets, hint)
            if len(bullets) >= limit:
                return bullets[:limit]
    for phrase in _iter_candidate_phrases(response):
        hint = _concept_to_hint(phrase)
        if hint:
            _push_unique(bullets, hint)
            if len(bullets) >= limit:
                return bullets[:limit]
    if len(bullets) < limit:
        for hint in _question_fallbacks(prompt):
            _push_unique(bullets, hint)
            if len(bullets) >= limit:
                break
    return bullets[:limit]


def _load_records(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() in {".jsonl", ".jsonl.gz"}:
        return [dict(record) for record in read_jsonl(path)]
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


def _load_model(config_path: Optional[Path]) -> tuple[NeuralSemanticModel, SemanticModelConfig]:
    config = load_config(config_path)
    model = NeuralSemanticModel(config=config)
    return model, config


def _run_generation(
    prompt: str,
    persona: Optional[str],
    config_path: Optional[Path],
    workspace: Path,
) -> None:
    """Load the model artifacts and emit a generated response."""

    model, _ = _load_model(config_path)
    artifacts_dir = Path(workspace)
    if (artifacts_dir / "embeddings.json").exists():
        model = NeuralSemanticModel.load(artifacts_dir, config=model.config)
    result = model.generate(normalise_text(prompt), persona=persona)
    typer.echo(result.response)


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
    config_path: Optional[Path] = CONFIG_OPTION,
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
    config_path: Optional[Path] = CONFIG_OPTION,
    workspace: Path = DIAG_WORKSPACE_OPTION,
    output: Optional[Path] = DIAG_OUTPUT_OPTION,
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
    prompt: str = typer.Argument(..., help="Prompt string; use '-' to read from standard input."),
    persona: Optional[str] = PERSONA_OPTION,
    config_path: Optional[Path] = GENERATE_CONFIG_OPTION,
    workspace: Path = GENERATE_WORKSPACE_OPTION,
) -> None:
    """Generate a persona-conditioned response."""

    if prompt == "-":
        prompt = sys.stdin.read()
    if not prompt.strip():
        raise typer.BadParameter("Empty prompt.")
    _run_generation(prompt, persona, config_path, workspace)


@app.command()
def clipboard(
    workspace: Path = CLIPBOARD_WORKSPACE_OPTION,
    persona: str = CLIPBOARD_PERSONA_OPTION,
    config_path: Optional[Path] = GENERATE_CONFIG_OPTION,
) -> None:
    """Generate a response using the current clipboard contents."""

    try:
        text = get_clipboard_text()
    except ClipboardError as exc:
        message = str(exc) or "Unable to read clipboard."
        typer.secho(message, err=True)
        raise typer.Exit(code=1) from exc

    if not text.strip():
        typer.secho("Clipboard is empty.", err=True)
        raise typer.Exit(code=1)

    _run_generation(text, persona, config_path, workspace)


@app.command("ask-tight")
def ask_tight(
    prompt: str = typer.Argument(..., help="Prompt to respond to."),
    persona: Optional[str] = PERSONA_OPTION,
    config_path: Optional[Path] = GENERATE_CONFIG_OPTION,
    workspace: Path = GENERATE_WORKSPACE_OPTION,
    bullets: int = BULLET_COUNT_OPTION,
) -> None:
    """Generate exactly N actionable bullet points suitable for tight prompts."""

    model, _ = _load_model(config_path)
    artifacts_dir = Path(workspace)
    if (artifacts_dir / "embeddings.json").exists():
        model = NeuralSemanticModel.load(artifacts_dir, config=model.config)
    result = model.generate(normalise_text(prompt), persona=persona)
    knowledge_concepts: Iterable[str] = []
    if result.knowledge_selection is not None and result.knowledge_selection.concepts:
        knowledge_concepts = result.knowledge_selection.concepts
    bullets_out = _tight_bullet_points(prompt, result.response, knowledge_concepts, bullets)
    for bullet in bullets_out:
        typer.echo(f"• {bullet}")


@app.command()
def knowledge(
    prompt: str = typer.Argument(..., help="Prompt to analyse."),
    persona: Optional[str] = PERSONA_OPTION,
    config_path: Optional[Path] = GENERATE_CONFIG_OPTION,
    workspace: Path = GENERATE_WORKSPACE_OPTION,
) -> None:
    """Inspect the knowledge selection for a prompt."""

    model, _ = _load_model(config_path)
    artifacts_dir = Path(workspace)
    if (artifacts_dir / "embeddings.json").exists():
        model = NeuralSemanticModel.load(artifacts_dir, config=model.config)
    result = model.generate(normalise_text(prompt), persona=persona)
    selection = result.knowledge_selection
    if selection is None or not selection.concepts:
        payload = {
            "concepts": [],
            "relevance": 0.0,
            "coverage": 0.0,
            "cohesion": 0.0,
            "collaboration": 0.0,
            "diversity": 0.0,
            "knowledge_raw": 0.0,
            "gate_mean": 0.0,
        }
    else:
        payload = {
            "concepts": list(selection.concepts),
            "relevance": selection.relevance,
            "coverage": selection.coverage,
            "cohesion": selection.cohesion,
            "collaboration": selection.collaboration,
            "diversity": selection.diversity,
            "knowledge_raw": selection.knowledge_raw,
            "gate_mean": selection.gate_mean,
        }
    typer.echo(json.dumps(payload, indent=2))


@app.command("compliance-report")
def compliance_report(
    payload_path: Path = COMPLIANCE_PAYLOAD_ARGUMENT,
    json_output: Path = COMPLIANCE_JSON_OUTPUT_OPTION,
    markdown_output: Path = COMPLIANCE_MARKDOWN_OUTPUT_OPTION,
) -> None:
    """Generate Markdown and JSON compliance reports from a payload."""

    summary, cases = load_report_payload(payload_path)
    json_output = Path(json_output)
    markdown_output = Path(markdown_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    write_reports(
        summary,
        cases,
        json_path=json_output,
        markdown_path=markdown_output,
    )
    typer.echo(
        f"Wrote compliance report to {markdown_output} and {json_output}",
    )


if __name__ == "__main__":
    app()

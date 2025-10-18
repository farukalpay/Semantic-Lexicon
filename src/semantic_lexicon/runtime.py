"""Runtime helpers for lightweight text generation.

This module exposes a small ``run`` helper that mirrors the behaviour of the
``semantic-lexicon`` CLI while being resilient to partially configured
workspaces.  The helper is intentionally defensive: it serves literal
directives ("return only the number 7"), handles the structured matrix markdown
prompt used in the quickstart script, and falls back to a neutral message if
the neural model cannot be loaded or has not been trained yet.

The goal is to provide an ergonomic entry-point for scripts that expect the
library to "just work" even when users skip prerequisite steps such as
training.  We therefore cache models per workspace, catch common failure modes
(``ValueError`` raised by an untrained classifier), and make sure to always
return a string response instead of propagating exceptions.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Optional

from .config import SemanticModelConfig, load_config
from .generator import (
    _maybe_generate_literal_response,
    _maybe_generate_structured_matrix_response,
)
from .logging import configure_logging
from .model import NeuralSemanticModel

LOGGER = configure_logging(logger_name=__name__)

_DEFAULT_WORKSPACE = Path("artifacts")


def _resolve_workspace(workspace: Optional[Path]) -> Path:
    if workspace is None:
        return _DEFAULT_WORKSPACE
    return Path(workspace)


@cache
def _load_model(workspace: str) -> NeuralSemanticModel:
    """Load (and cache) a ``NeuralSemanticModel`` for ``workspace``.

    The loader is intentionally forgiving: if the workspace does not contain
    trained artifacts we simply return a fresh model configured with defaults.
    The caller is expected to handle potential ``ValueError`` raised when the
    untrained model is used for generation.
    """

    config: SemanticModelConfig = load_config(None)
    model = NeuralSemanticModel(config=config)
    directory = Path(workspace)
    embeddings_path = directory / "embeddings.json"
    if embeddings_path.exists():
        LOGGER.info("Loaded trained artifacts from %s", directory)
        model = NeuralSemanticModel.load(directory, config=model.config)
    else:
        LOGGER.warning(
            "Workspace %s does not contain trained artifacts; using cold model",
            directory,
        )
    return model


def _fallback_response(prompt: str, persona: Optional[str]) -> str:
    """Build a neutral fallback message for unhandled prompts."""

    prompt_text = prompt.strip()
    if not prompt_text:
        prompt_text = "this request"
    persona_prefix = "Let's work through this together."
    if persona:
        persona_lower = persona.lower()
        if persona_lower == "analyst":
            persona_prefix = "Analyst note:"
        elif persona_lower == "tutor":
            persona_prefix = "Tutor tip:"
        else:
            persona_prefix = f"{persona.capitalize()} insight:"
    return f"{persona_prefix} {prompt_text}".strip()


def _preflight_response(prompt: str) -> Optional[str]:
    """Return a deterministic response for literal prompts when possible."""

    structured = _maybe_generate_structured_matrix_response(prompt)
    if structured is not None:
        return structured
    literal = _maybe_generate_literal_response(prompt)
    if literal is not None:
        return literal
    return None


def run(
    prompt: str,
    persona: Optional[str] = None,
    *,
    workspace: Optional[Path] = None,
) -> str:
    """Generate a response for ``prompt`` with optional ``persona``.

    The helper first checks whether the prompt requests a literal or structured
    answer.  If not, it attempts to load a cached model from ``workspace``
    (defaulting to ``artifacts``).  Any failure to use the model results in a
    graceful fallback message so callers never have to handle exceptions.
    """

    prompt_text = str(prompt or "")

    deterministic = _preflight_response(prompt_text)
    if deterministic is not None:
        return deterministic

    workspace_path = _resolve_workspace(workspace)
    model = _load_model(str(workspace_path.resolve()))
    try:
        result = model.generate(prompt_text, persona=persona)
        return result.response
    except ValueError as exc:
        LOGGER.warning("Model unavailable for generation: %s", exc)
    except Exception:  # pragma: no cover - defensive logging
        LOGGER.exception("Generation failed; returning fallback response")

    # Either the model is untrained or another error occurred.  Fall back to a
    # neutral but persona-aware message so callers receive a deterministic
    # string regardless of environment state.
    return _fallback_response(prompt_text, persona)


__all__ = ["run"]

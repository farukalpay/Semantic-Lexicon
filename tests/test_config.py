# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

from __future__ import annotations

from pathlib import Path

from semantic_lexicon.config import SemanticModelConfig, load_config


def test_load_config_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("intent:\n  epochs: 2\n")
    config = load_config(path)
    assert isinstance(config, SemanticModelConfig)
    assert config.intent.epochs == 2
    assert config.embeddings.dimension == 50

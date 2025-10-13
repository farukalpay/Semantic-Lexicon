# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

from __future__ import annotations

from semantic_lexicon.diagnostics import DiagnosticsSuite
from semantic_lexicon.model import NeuralSemanticModel
from semantic_lexicon.training import Trainer, TrainerConfig


def test_diagnostics_runs(workspace) -> None:
    model = NeuralSemanticModel()
    trainer = Trainer(model, TrainerConfig())
    trainer.train()
    suite = DiagnosticsSuite(model=model)
    result = suite.run()
    assert result.embedding_stats
    assert result.generations

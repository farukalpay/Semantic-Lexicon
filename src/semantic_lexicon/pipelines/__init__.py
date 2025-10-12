"""Predefined training and evaluation pipelines."""

from __future__ import annotations

from pathlib import Path

from ..config import load_config
from ..model import NeuralSemanticModel
from ..training import Trainer, TrainerConfig


def prepare_and_train(
    config_path: Path | None = None,
    workspace: Path = Path("artifacts"),
) -> NeuralSemanticModel:
    """Load configuration, train the model, and return it."""

    config = load_config(config_path)
    model = NeuralSemanticModel(config=config)
    trainer = Trainer(
        model,
        TrainerConfig(
            workspace=workspace,
            intent_dataset=workspace / "intent.jsonl",
            knowledge_dataset=workspace / "knowledge.jsonl",
        ),
    )
    trainer.train()
    return trainer.model

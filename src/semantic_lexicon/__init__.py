"""Semantic Lexicon package."""

from .config import SemanticModelConfig, load_config
from .model import NeuralSemanticModel
from .training import Trainer, TrainerConfig

__all__ = [
    "SemanticModelConfig",
    "TrainerConfig",
    "NeuralSemanticModel",
    "Trainer",
    "load_config",
]

__version__ = "0.1.0"

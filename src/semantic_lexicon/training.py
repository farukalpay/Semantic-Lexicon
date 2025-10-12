"""Training helpers for the Semantic Lexicon model."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .config import SemanticModelConfig
from .diagnostics import DiagnosticsResult, DiagnosticsSuite
from .intent import IntentExample
from .knowledge import KnowledgeEdge
from .logging import configure_logging
from .model import NeuralSemanticModel
from .utils import normalise_text, read_jsonl, seed_everything, write_jsonl

LOGGER = configure_logging(logger_name=__name__)


@dataclass
class TrainerConfig:
    """Configuration for the training pipeline."""

    workspace: Path = Path("artifacts")
    intent_dataset: Path = Path("src/semantic_lexicon/data/intent.jsonl")
    knowledge_dataset: Path = Path("src/semantic_lexicon/data/knowledge.jsonl")
    seed: int = 0


class Trainer:
    """High level training pipeline."""

    def __init__(self, model: NeuralSemanticModel, config: TrainerConfig | None = None) -> None:
        self.model = model
        self.config = config or TrainerConfig()

    # Corpus preparation -----------------------------------------------------------
    def prepare_corpus(self, raw_intents: Iterable[dict], raw_knowledge: Iterable[dict]) -> None:
        workspace = Path(self.config.workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        intent_path = workspace / "intent.jsonl"
        knowledge_path = workspace / "knowledge.jsonl"
        processed_intents = [
            {
                "text": normalise_text(item["text"]),
                "intent": item["intent"],
            }
            for item in raw_intents
        ]
        processed_knowledge = [
            {
                "head": normalise_text(item["head"]),
                "relation": item["relation"],
                "tail": normalise_text(item["tail"]),
            }
            for item in raw_knowledge
        ]
        write_jsonl(intent_path, processed_intents)
        write_jsonl(knowledge_path, processed_knowledge)
        LOGGER.info("Prepared corpus under %s", workspace)
        self.config.intent_dataset = intent_path
        self.config.knowledge_dataset = knowledge_path

    # Training --------------------------------------------------------------------
    def train(self) -> None:
        seed_everything(self.config.seed)
        intents = self._load_intent_examples(Path(self.config.intent_dataset))
        knowledge = self._load_knowledge_edges(Path(self.config.knowledge_dataset))
        self.model.train_intents(intents)
        self.model.train_knowledge(knowledge)

    def _load_intent_examples(self, path: Path) -> list[IntentExample]:
        dataset = []
        for record in read_jsonl(path):
            dataset.append(IntentExample(text=str(record["text"]), intent=str(record["intent"])))
        return dataset

    def _load_knowledge_edges(self, path: Path) -> list[KnowledgeEdge]:
        dataset = []
        for record in read_jsonl(path):
            dataset.append(
                KnowledgeEdge(
                    head=str(record["head"]),
                    relation=str(record["relation"]),
                    tail=str(record["tail"]),
                )
            )
        return dataset

    # Diagnostics -----------------------------------------------------------------
    def run_diagnostics(self) -> DiagnosticsResult:
        suite = DiagnosticsSuite(model=self.model)
        return suite.run()


def train_from_config(
    config: SemanticModelConfig,
    trainer_config: TrainerConfig | None = None,
) -> NeuralSemanticModel:
    """Convenience helper to instantiate and train a model from configuration."""

    trainer_config = trainer_config or TrainerConfig()
    model = NeuralSemanticModel(config=config)
    trainer = Trainer(model, trainer_config)
    trainer.train()
    return model

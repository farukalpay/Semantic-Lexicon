# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""Training helpers for the Semantic Lexicon model."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

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

    def __init__(self, model: NeuralSemanticModel, config: Optional[TrainerConfig] = None) -> None:
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
        processed_knowledge: list[dict[str, str]] = []
        for item in raw_knowledge:
            # Accept both triple-style (head/relation/tail) and simplified (key/text) records.
            if "head" in item or "tail" in item:
                head_raw = item.get("head")
                tail_raw = item.get("tail")
                relation_raw = item.get("relation")
                default_relation = "related_to"
            elif "key" in item and "text" in item:
                head_raw = item.get("key")
                tail_raw = item.get("text")
                relation_raw = item.get("relation")
                default_relation = "describes"
            else:
                msg = "Knowledge record must provide ('head','tail') or ('key','text') fields"
                raise KeyError(msg)
            if not isinstance(head_raw, str) or not head_raw.strip():
                raise TypeError("Knowledge 'head'/'key' field must be a non-empty string")
            if not isinstance(tail_raw, str) or not tail_raw.strip():
                raise TypeError("Knowledge 'tail'/'text' field must be a non-empty string")
            if isinstance(relation_raw, str) and relation_raw.strip():
                relation_value = relation_raw
            else:
                relation_value = default_relation
            processed_knowledge.append(
                {
                    "head": normalise_text(head_raw),
                    "relation": normalise_text(relation_value),
                    "tail": normalise_text(tail_raw),
                }
            )
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
        dataset: list[IntentExample] = []
        for record in read_jsonl(path):
            mapping = cast(Mapping[str, object], record)
            text = _require_str(mapping, "text")
            intent = _require_str(mapping, "intent")
            feedback = _coerce_numeric(mapping.get("feedback", 0.95), 0.95)
            dataset.append(IntentExample(text=text, intent=intent, feedback=feedback))
        return dataset

    def _load_knowledge_edges(self, path: Path) -> list[KnowledgeEdge]:
        dataset: list[KnowledgeEdge] = []
        for record in read_jsonl(path):
            mapping = cast(Mapping[str, object], record)
            head = _require_str(mapping, "head")
            relation = _require_str(mapping, "relation")
            tail = _require_str(mapping, "tail")
            dataset.append(KnowledgeEdge(head=head, relation=relation, tail=tail))
        return dataset

    # Diagnostics -----------------------------------------------------------------
    def run_diagnostics(self) -> DiagnosticsResult:
        suite = DiagnosticsSuite(model=self.model)
        return suite.run()


def train_from_config(
    config: SemanticModelConfig,
    trainer_config: Optional[TrainerConfig] = None,
) -> NeuralSemanticModel:
    """Convenience helper to instantiate and train a model from configuration."""

    trainer_config = trainer_config or TrainerConfig()
    model = NeuralSemanticModel(config=config)
    trainer = Trainer(model, trainer_config)
    trainer.train()
    return model


def _require_str(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if isinstance(value, str):
        return value
    raise TypeError(f"Expected string for '{key}', received {type(value)!r}")


def _coerce_numeric(value: object, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:  # pragma: no cover - defensive
            raise TypeError("Numeric field must be parseable as float") from exc
    if value is None:
        return default
    raise TypeError(f"Numeric field must be float-compatible, received {type(value)!r}")

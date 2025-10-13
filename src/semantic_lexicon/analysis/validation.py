# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""Validation helpers for cross-domain intent evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from ..utils import read_jsonl

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    from ..intent import IntentClassifier


@dataclass(frozen=True)
class ValidationRecord:
    """Single labelled prompt used during validation."""

    text: str
    intent: str
    domain: str
    feedback: float


@dataclass(frozen=True)
class PredictionSummary:
    """Summary of a classifier prediction for reporting."""

    text: str
    domain: str
    expected: str
    predicted: str
    confidence: float
    reward: float


@dataclass(frozen=True)
class ValidationMetrics:
    """Aggregated validation statistics."""

    accuracy: float
    reward_summary: dict[str, float]
    ece_before: float
    ece_after: float
    ece_reduction: float
    domain_accuracy: dict[str, float]
    predictions: Sequence[PredictionSummary]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["predictions"] = [asdict(pred) for pred in self.predictions]
        return data


def _default_validation_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "cross_domain_validation.jsonl"


def _coerce_to_str(value: object, field: str, *, default: str | None = None) -> str:
    if isinstance(value, str):
        return value
    if value is None and default is not None:
        return default
    msg = f"Expected string for '{field}', received {type(value)!r}"
    raise TypeError(msg)


def _coerce_feedback(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:  # pragma: no cover - defensive
            raise TypeError("Feedback must be numeric") from exc
    if value is None:
        return 0.9
    raise TypeError(f"Feedback must be numeric, received {type(value)!r}")


def load_validation_records(path: Path | None = None) -> list[ValidationRecord]:
    """Load labelled prompts for evaluation."""

    path = path or _default_validation_path()
    records: list[ValidationRecord] = []
    for raw in read_jsonl(path):
        mapping = cast(Mapping[str, object], raw)
        records.append(
            ValidationRecord(
                text=_coerce_to_str(mapping.get("text"), "text"),
                intent=_coerce_to_str(mapping.get("intent"), "intent"),
                domain=_coerce_to_str(
                    mapping.get("domain"),
                    "domain",
                    default="unknown",
                ),
                feedback=_coerce_feedback(mapping.get("feedback", 0.9)),
            )
        )
    if not records:
        msg = f"No validation records found at {path}"
        raise ValueError(msg)
    return records


def evaluate_classifier(
    classifier: IntentClassifier,
    records: Sequence[ValidationRecord],
    *,
    reward_threshold: float = 0.7,
) -> ValidationMetrics:
    """Evaluate ``classifier`` and return aggregate metrics."""

    raw_ece, calibrated_ece = classifier.calibration_report
    total = len(records)
    correct = 0
    rewards: list[float] = []
    predictions: list[PredictionSummary] = []
    domain_hits: dict[str, list[bool]] = {}
    for record in records:
        probabilities: dict[str, float] = classifier.predict_proba(record.text)
        predicted_intent = max(probabilities.items(), key=lambda item: item[1])[0]
        reward = float(
            classifier.reward(
                record.text,
                predicted_intent,
                record.intent,
                record.feedback,
            )
        )
        rewards.append(reward)
        predictions.append(
            PredictionSummary(
                text=record.text,
                domain=record.domain,
                expected=record.intent,
                predicted=predicted_intent,
                confidence=float(probabilities[predicted_intent]),
                reward=reward,
            )
        )
        is_correct = predicted_intent == record.intent
        correct += int(is_correct)
        domain_hits.setdefault(record.domain, []).append(is_correct)
        if reward < reward_threshold:
            msg = (
                f"Reward {reward:.2f} below threshold for domain='{record.domain}'"
                f" prompt='{record.text}'"
            )
            raise AssertionError(msg)
    accuracy = correct / total
    reward_array = np.asarray(rewards, dtype=float)
    quantiles = np.percentile(reward_array, [10, 25, 50, 75, 90])
    reward_summary = {
        "min": float(reward_array.min()),
        "max": float(reward_array.max()),
        "mean": float(reward_array.mean()),
        "p10": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
    }
    domain_accuracy = {domain: float(np.mean(flags)) for domain, flags in domain_hits.items()}
    ece_reduction = 1.0 if raw_ece == 0 else 1.0 - (calibrated_ece / raw_ece)
    return ValidationMetrics(
        accuracy=float(accuracy),
        reward_summary=reward_summary,
        ece_before=float(raw_ece),
        ece_after=float(calibrated_ece),
        ece_reduction=float(ece_reduction),
        domain_accuracy=domain_accuracy,
        predictions=predictions,
    )


def write_validation_report(metrics: ValidationMetrics, path: Path) -> None:
    """Persist validation metrics to ``path`` as JSON."""

    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as handle:
        json.dump(metrics.to_dict(), handle, indent=2)
        handle.write("\n")

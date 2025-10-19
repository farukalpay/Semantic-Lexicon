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
class DomainHierarchy:
    """Hierarchical metadata describing a validation domain."""

    domain: str
    path: tuple[str, ...]
    members: tuple[str, ...]
    traits: tuple[str, ...]
    lineage_traits: Mapping[tuple[str, ...], tuple[str, ...]]


@dataclass(frozen=True)
class ValidationRecord:
    """Single labelled prompt used during validation."""

    text: str
    intent: str
    domain: str
    feedback: float
    domain_path: tuple[str, ...]
    traits: tuple[str, ...]


@dataclass(frozen=True)
class PredictionSummary:
    """Summary of a classifier prediction for reporting."""

    text: str
    domain: str
    domain_path: tuple[str, ...]
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
    hierarchy_accuracy: dict[str, float]
    trait_accuracy: dict[str, float]
    predictions: Sequence[PredictionSummary]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["predictions"] = [asdict(pred) for pred in self.predictions]
        return data


def _default_validation_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "cross_domain_validation.jsonl"


def _default_domain_hierarchy_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "domain_hierarchy.jsonl"


def _coerce_to_str(value: object, field: str, *, default: str | None = None) -> str:
    if isinstance(value, str):
        return value
    if value is None and default is not None:
        return default
    msg = f"Expected string for '{field}', received {type(value)!r}"
    raise TypeError(msg)


def _coerce_optional_str(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _normalise_identifier(_coerce_to_str(value, field), field)


def _coerce_to_sequence_of_str(
    value: object,
    field: str,
    *,
    default: Sequence[str] | None = None,
) -> tuple[str, ...]:
    if value is None:
        return tuple(default or ())
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        result: list[str] = []
        for item in value:
            if not isinstance(item, str):
                msg = f"Expected all entries in '{field}' to be strings"
                raise TypeError(msg)
            result.append(item)
        return tuple(result)
    msg = f"Expected sequence of strings for '{field}', received {type(value)!r}"
    raise TypeError(msg)


def _merge_ordered_traits(*sources: Sequence[str]) -> tuple[str, ...]:
    seen: dict[str, str] = {}
    ordered: list[str] = []
    for values in sources:
        for value in values:
            key = value.casefold()
            if key not in seen:
                seen[key] = value
                ordered.append(value)
    return tuple(ordered)


def _normalise_identifier(value: str, field: str) -> str:
    normalised = value.strip()
    if not normalised:
        raise ValueError(f"'{field}' must be a non-empty string")
    return normalised


def _coerce_identifier_sequence(
    value: object,
    field: str,
    *,
    default: Sequence[str] | None = None,
) -> tuple[str, ...]:
    raw_values = _coerce_to_sequence_of_str(value, field, default=default)
    return tuple(_normalise_identifier(item, field) for item in raw_values)


def _coerce_lineage_prefix(
    raw_prefix: object,
    *,
    domain: str,
    path: Sequence[str],
) -> tuple[str, ...]:
    if not isinstance(raw_prefix, str):
        msg = f"Lineage trait keys for domain '{domain}' must be strings"
        raise TypeError(msg)
    parts = [segment.strip() for segment in raw_prefix.split(">") if segment.strip()]
    if not parts:
        msg = f"Lineage trait prefix for domain '{domain}' must be non-empty"
        raise ValueError(msg)
    prefix = tuple(_normalise_identifier(part, "lineage_traits") for part in parts)
    if len(prefix) > len(path):
        msg = (
            "Lineage trait prefix "
            f"'{raw_prefix}' for domain '{domain}' is deeper than the domain path"
        )
        raise ValueError(msg)
    path_prefix = tuple(entry.casefold() for entry in path[: len(prefix)])
    prefix_key = tuple(entry.casefold() for entry in prefix)
    if path_prefix != prefix_key:
        msg = (
            "Lineage trait prefix "
            f"'{raw_prefix}' for domain '{domain}' must align with the declared path"
        )
        raise ValueError(msg)
    return prefix


def _find_casefold_duplicates(values: Sequence[str]) -> list[str]:
    buckets: dict[str, list[str]] = {}
    for entry in values:
        key = entry.casefold()
        buckets.setdefault(key, []).append(entry)
    duplicates: list[str] = []
    for bucket in buckets.values():
        if len(bucket) > 1:
            duplicates.extend(sorted(set(bucket)))
    return sorted(duplicates, key=str.lower)


def load_domain_hierarchy(path: Path | None = None) -> dict[str, DomainHierarchy]:
    """Load domain hierarchy metadata indexed by domain name."""

    path = path or _default_domain_hierarchy_path()
    hierarchy: dict[str, DomainHierarchy] = {}
    if not Path(path).exists():
        return hierarchy
    canonical_domains: dict[str, str] = {}
    member_domains: dict[str, str] = {}
    for raw in read_jsonl(Path(path)):
        mapping = cast(Mapping[str, object], raw)
        domain = _normalise_identifier(_coerce_to_str(mapping.get("domain"), "domain"), "domain")
        domain_key = domain.casefold()
        existing_domain = canonical_domains.get(domain_key)
        if existing_domain:
            if existing_domain == domain:
                msg = f"Duplicate hierarchy definition for domain '{domain}'"
            else:
                msg = (
                    f"Domain '{domain}' conflicts with previously declared domain "
                    f"'{existing_domain}'"
                )
            raise ValueError(msg)
        canonical_domains[domain_key] = domain
        path_values = _coerce_identifier_sequence(
            mapping.get("path"),
            "path",
            default=(domain,),
        )
        if not path_values:
            path_values = (domain,)
        duplicates = _find_casefold_duplicates(path_values)
        if duplicates:
            dup = ", ".join(duplicates)
            msg = f"Duplicate path entry(ies) {dup} declared for domain '{domain}'"
            raise ValueError(msg)
        if path_values[-1].casefold() != domain.casefold():
            msg = f"Hierarchy path for domain '{domain}' must end with the domain name"
            raise ValueError(msg)
        members = _coerce_identifier_sequence(mapping.get("members"), "members")
        duplicates = _find_casefold_duplicates(members)
        if duplicates:
            dup = ", ".join(duplicates)
            msg = f"Duplicate member identifier(s) {dup} declared for domain '{domain}'"
            raise ValueError(msg)
        traits = _coerce_identifier_sequence(mapping.get("traits"), "traits")
        duplicates = _find_casefold_duplicates(traits)
        if duplicates:
            dup = ", ".join(duplicates)
            msg = f"Duplicate trait identifier(s) {dup} declared for domain '{domain}'"
            raise ValueError(msg)
        lineage_traits_field = mapping.get("lineage_traits")
        lineage_traits: dict[tuple[str, ...], tuple[str, ...]] = {}
        if lineage_traits_field is not None:
            if not isinstance(lineage_traits_field, Mapping):
                msg = (
                    "Expected mapping for 'lineage_traits' in domain "
                    f"'{domain}', received {type(lineage_traits_field)!r}"
                )
                raise TypeError(msg)
            for raw_prefix, raw_values in lineage_traits_field.items():
                prefix = _coerce_lineage_prefix(
                    raw_prefix,
                    domain=domain,
                    path=path_values,
                )
                if prefix in lineage_traits:
                    msg = (
                        "Duplicate lineage trait prefix "
                        f"'{raw_prefix}' declared for domain '{domain}'"
                    )
                    raise ValueError(msg)
                trait_values = _coerce_identifier_sequence(
                    raw_values,
                    f"lineage_traits[{raw_prefix!r}]",
                )
                duplicates = _find_casefold_duplicates(trait_values)
                if duplicates:
                    dup = ", ".join(duplicates)
                    msg = (
                        "Duplicate trait identifier(s) "
                        f"{dup} declared for lineage prefix '{raw_prefix}' in domain '{domain}'"
                    )
                    raise ValueError(msg)
                lineage_traits[prefix] = trait_values
        for member in members:
            member_key = member.casefold()
            owning_domain = canonical_domains.get(member_key)
            if owning_domain:
                msg = (
                    f"Member '{member}' for domain '{domain}' conflicts with domain "
                    f"identifier '{owning_domain}'"
                )
                raise ValueError(msg)
            existing = member_domains.get(member_key)
            if existing and existing != domain:
                msg = (
                    f"Contradictory member '{member}' declared for domains "
                    f"'{existing}' and '{domain}'"
                )
                raise ValueError(msg)
            member_domains[member_key] = domain
        hierarchy[domain] = DomainHierarchy(
            domain=domain,
            path=path_values,
            members=members,
            traits=traits,
            lineage_traits=lineage_traits,
        )
    return hierarchy


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


def load_validation_records(
    path: Path | None = None,
    *,
    hierarchy_path: Path | None = None,
) -> list[ValidationRecord]:
    """Load labelled prompts for evaluation."""

    path = path or _default_validation_path()
    hierarchy_index = load_domain_hierarchy(hierarchy_path)
    records: list[ValidationRecord] = []
    for raw in read_jsonl(path):
        mapping = cast(Mapping[str, object], raw)
        domain = _coerce_to_str(mapping.get("domain"), "domain", default="unknown")
        hierarchy = hierarchy_index.get(domain)
        traits: tuple[str, ...]
        domain_path: tuple[str, ...]
        if hierarchy:
            domain_path = hierarchy.path
            trait_sources = [
                hierarchy.lineage_traits.get(hierarchy.path[:depth], ())
                for depth in range(1, len(hierarchy.path) + 1)
            ]
            trait_sources.append(hierarchy.traits)
            traits = _merge_ordered_traits(*trait_sources)
        else:
            domain_path = (domain,)
            traits = ()
        member = _coerce_optional_str(mapping.get("domain_member"), "domain_member")
        if member:
            if not hierarchy:
                msg = (
                    f"Validation record references member '{member}' for domain '{domain}'"
                    " but no hierarchy metadata is available"
                )
                raise ValueError(msg)
            domain_path = domain_path + (member,)
            if hierarchy and hierarchy.members and member not in hierarchy.members:
                msg = (
                    f"Validation record references unknown member '{member}' for domain '{domain}'"
                )
                raise ValueError(msg)
        records.append(
            ValidationRecord(
                text=_coerce_to_str(mapping.get("text"), "text"),
                intent=_coerce_to_str(mapping.get("intent"), "intent"),
                domain=domain,
                feedback=_coerce_feedback(mapping.get("feedback", 0.9)),
                domain_path=domain_path,
                traits=traits,
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
    hierarchy_hits: dict[tuple[str, ...], list[bool]] = {}
    trait_hits: dict[str, list[bool]] = {}
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
                domain_path=record.domain_path,
                expected=record.intent,
                predicted=predicted_intent,
                confidence=float(probabilities[predicted_intent]),
                reward=reward,
            )
        )
        is_correct = predicted_intent == record.intent
        correct += int(is_correct)
        domain_hits.setdefault(record.domain, []).append(is_correct)
        for depth in range(1, len(record.domain_path) + 1):
            prefix = record.domain_path[:depth]
            hierarchy_hits.setdefault(prefix, []).append(is_correct)
        for trait in record.traits:
            trait_hits.setdefault(trait, []).append(is_correct)
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
    hierarchy_accuracy = {
        " > ".join(path): float(np.mean(flags)) for path, flags in hierarchy_hits.items()
    }
    trait_accuracy = {trait: float(np.mean(flags)) for trait, flags in trait_hits.items()}
    ece_reduction = 1.0 if raw_ece == 0 else 1.0 - (calibrated_ece / raw_ece)
    return ValidationMetrics(
        accuracy=float(accuracy),
        reward_summary=reward_summary,
        ece_before=float(raw_ece),
        ece_after=float(calibrated_ece),
        ece_reduction=float(ece_reduction),
        domain_accuracy=domain_accuracy,
        hierarchy_accuracy=hierarchy_accuracy,
        trait_accuracy=trait_accuracy,
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

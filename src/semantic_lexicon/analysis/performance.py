# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""Performance and profiling utilities for the intent pipeline."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..intent import IntentClassifier


@dataclass(frozen=True)
class PerformanceReport:
    """Summary of baseline vs optimised inference efficiency."""

    baseline_latency_ms: float
    optimised_latency_ms: float
    latency_improvement_pct: float
    baseline_memory_kb: float
    optimised_memory_kb: float
    memory_reduction_pct: float

    def to_dict(self) -> dict[str, float]:
        return {
            "baseline_latency_ms": self.baseline_latency_ms,
            "optimised_latency_ms": self.optimised_latency_ms,
            "latency_improvement_pct": self.latency_improvement_pct,
            "baseline_memory_kb": self.baseline_memory_kb,
            "optimised_memory_kb": self.optimised_memory_kb,
            "memory_reduction_pct": self.memory_reduction_pct,
        }


def _estimate_memory_kb(classifier: IntentClassifier) -> float:
    weights = getattr(classifier, "_weights_for_inference", None)
    cache = getattr(classifier, "_vector_cache", None)
    total_bytes = 0
    if weights is not None:
        total_bytes += int(np.asarray(weights).nbytes)
    if cache:
        for entry in cache.values():
            if isinstance(entry, tuple):
                vector = entry[0]
            else:
                vector = entry
            total_bytes += int(np.asarray(vector).nbytes)
    return total_bytes / 1024.0


def benchmark_inference(
    baseline: IntentClassifier,
    optimised: IntentClassifier,
    prompts: Sequence[str],
    *,
    repeat: int = 3,
    warmup: int = 1,
) -> PerformanceReport:
    """Benchmark baseline vs optimised classifiers on ``prompts``."""

    baseline.set_cache_enabled(False)
    optimised.set_cache_enabled(True)
    if not prompts:
        expanded: list[str] = []
    else:
        expanded = list(prompts) * max(repeat, 1)

    if expanded:
        start = time.perf_counter()
        for text in expanded:
            baseline.predict_proba(text)
        baseline_elapsed = time.perf_counter() - start
        for _ in range(max(warmup, 1)):
            for text in expanded:
                optimised.predict_proba(text)
        start = time.perf_counter()
        for text in expanded:
            optimised.predict_proba(text)
        optimised_elapsed = time.perf_counter() - start
        baseline_latency = (baseline_elapsed / len(expanded)) * 1000.0
        optimised_latency = (optimised_elapsed / len(expanded)) * 1000.0
    else:
        baseline_latency = 0.0
        optimised_latency = 0.0
    improvement = 0.0
    if baseline_latency > 0:
        improvement = ((baseline_latency - optimised_latency) / baseline_latency) * 100.0
    baseline_mem = _estimate_memory_kb(baseline)
    optimised_mem = _estimate_memory_kb(optimised)
    reduction = 0.0
    if baseline_mem > 0:
        reduction = ((baseline_mem - optimised_mem) / baseline_mem) * 100.0
    return PerformanceReport(
        baseline_latency_ms=float(baseline_latency),
        optimised_latency_ms=float(optimised_latency),
        latency_improvement_pct=float(improvement),
        baseline_memory_kb=float(baseline_mem),
        optimised_memory_kb=float(optimised_mem),
        memory_reduction_pct=float(reduction),
    )

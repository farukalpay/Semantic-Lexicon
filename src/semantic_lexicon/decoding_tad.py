# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional, Protocol, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .oracle import Oracle, OracleReport


FloatArray = NDArray[np.float64]

# ---------- Public contracts ----------


class ModelLike(Protocol):
    @property
    def vocab(self) -> Sequence[str]: ...

    def next_logits(self, prefix_token_ids: Sequence[int]) -> np.ndarray: ...

    def eos_id(self) -> int: ...


class DecodeAbortReason(str, Enum):
    NONE = "none"
    ABSTAIN_LOW_SAFE_MASS = "abstain_low_safe_mass"
    ORACLE_BLOCKED_ALL = "oracle_blocked_all"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class TADConfig:
    tau: float = 0.15
    max_new_tokens: int = 64
    abstain_token: Optional[int] = None
    enable_backoff: bool = False
    backoff_attempts: int = 0
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    allow_eos_escape: bool = True
    min_new_tokens: int = 0
    max_decode_ms: Optional[int] = None
    stop_token_ids: tuple[int, ...] = ()
    no_repeat_ngram: int = 0
    repetition_penalty: float = 1.0
    enable_step_logs: bool = True
    log_top_k: int = 5
    preselect_top_k: int = 0
    preselect_top_p: float = 1.0

    def validate(self) -> None:
        if not (0.0 <= self.tau <= 1.0):
            raise ValueError("tau must be in [0,1]")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
        if self.temperature < 0.0:
            raise ValueError("temperature must be >= 0")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in (0,1]")
        if not (0.0 < self.preselect_top_p <= 1.0):
            raise ValueError("preselect_top_p must be in (0,1]")
        if self.top_k < 0 or self.preselect_top_k < 0:
            raise ValueError("top_k values must be >= 0")
        if self.no_repeat_ngram < 0:
            raise ValueError("no_repeat_ngram must be >= 0")
        if self.repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be > 0")


@dataclass
class TADStepLog:
    t: int
    pi_safe: float
    picked_id: Optional[int]
    blocked_count: int
    reasons_for_picked: list[str] = field(default_factory=list)
    topk_ids: list[int] = field(default_factory=list)
    topk_probs: list[float] = field(default_factory=list)
    aborted: bool = False


@dataclass
class TADOutcome:
    token_ids: list[int]
    abstained: bool
    abort_reason: DecodeAbortReason
    logs: list[TADStepLog]


# ---------- Exceptions for control flow ----------


class OracleContractError(RuntimeError): ...


class DecodeTimeout(RuntimeError): ...


class NumericsError(RuntimeError): ...


# ---------- Helpers ----------


def _safe_softmax(x: np.ndarray) -> FloatArray:
    if not np.isfinite(x).all():
        raise NumericsError("logits contain NaN/Inf")
    x64 = x.astype(np.float64, copy=False)
    m = np.max(x64)
    if not np.isfinite(m):
        raise NumericsError("max(logits) is not finite")
    z = x64 - m
    e = np.exp(z, dtype=np.float64)
    s = e.sum()
    if s <= 0.0 or not np.isfinite(s):
        return cast("FloatArray", np.full_like(x64, 1.0 / x64.size, dtype=np.float64))
    return cast("FloatArray", e / s)


def _apply_repetition_penalty(logits: np.ndarray, generated: Sequence[int], penalty: float) -> None:
    if penalty == 1.0 or not generated:
        return
    adj = math.log(penalty)
    for tid in set(generated):
        logits[tid] -= adj


def _apply_no_repeat_ngram(prefix: Sequence[int], next_mask: np.ndarray, n: int) -> None:
    if n <= 0 or len(prefix) < n - 1:
        return
    tail = tuple(prefix[-(n - 1) :])
    seen = set()
    for i in range(len(prefix) - n + 1):
        if tuple(prefix[i : i + n - 1]) == tail:
            seen.add(prefix[i + n - 1])
    if seen:
        next_mask[np.array(list(seen), dtype=int)] = False


def _trim_by_topk_topp(probs: np.ndarray, k: int, p: float) -> np.ndarray:
    idx = np.argsort(-probs)
    if k > 0:
        idx = idx[: min(k, idx.size)]
    if p < 1.0:
        cumulative = np.cumsum(probs[idx])
        keep = idx[cumulative <= p]
        if keep.size == 0:
            keep = idx[:1]
        idx = keep
    mask = np.zeros_like(probs, dtype=bool)
    mask[idx] = True
    return mask


def _choose_id(
    logits: np.ndarray,
    safe_mask: np.ndarray,
    temperature: float,
    top_p: float,
    top_k: int,
    rng: np.random.Generator,
) -> int:
    masked = np.full_like(logits, -np.inf, dtype=np.float64)
    masked[safe_mask] = logits[safe_mask]
    if temperature <= 0.0:
        return int(np.argmax(masked))
    probs = _safe_softmax(masked / max(temperature, 1e-6))
    shortlist = _trim_by_topk_topp(probs, top_k, top_p)
    final_mask = shortlist & safe_mask
    if not np.any(final_mask):
        final_mask = safe_mask
    probs = probs * final_mask
    total = probs.sum()
    if total <= 0.0:
        return int(np.argmax(masked))
    probs = probs / total
    return int(rng.choice(np.arange(probs.size), p=probs))


# ---------- Backoff interface (optional) ----------


class BackoffHandler(Protocol):
    def on_abstain(self, prefix_token_ids: list[int], pi_safe: float) -> tuple[list[int], bool]:
        """Return a new prefix and whether to retry decoding."""


# ---------- Main decode ----------


def truth_aware_decode(
    model: ModelLike,
    oracle: Oracle,
    prefix_token_ids: list[int],
    cfg: TADConfig | None = None,
    *,
    backoff: Optional[BackoffHandler] = None,
    rng: Optional[np.random.Generator] = None,
    now: Callable[[], float] = time.time,
) -> TADOutcome:
    cfg = cfg or TADConfig()
    cfg.validate()
    if rng is None:
        rng = np.random.default_rng()

    start_time = now()
    generated: list[int] = []
    logs: list[TADStepLog] = []
    attempts_left = max(0, cfg.backoff_attempts)

    while True:
        outcome = _decode_once(model, oracle, prefix_token_ids, cfg, rng, now, start_time)

        logs.extend(outcome.logs)
        generated.extend(outcome.token_ids)

        if not outcome.abstained:
            return TADOutcome(
                token_ids=generated,
                abstained=False,
                abort_reason=DecodeAbortReason.NONE,
                logs=logs,
            )

        abort_reason = outcome.abort_reason

        if (
            cfg.enable_backoff
            and attempts_left > 0
            and backoff is not None
            and abort_reason == DecodeAbortReason.ABSTAIN_LOW_SAFE_MASS
        ):
            attempts_left -= 1
            new_prefix, retry = backoff.on_abstain(prefix_token_ids, _last_pi_safe(logs))
            if retry:
                prefix_token_ids = new_prefix
                continue

        if cfg.abstain_token is not None:
            generated.append(cfg.abstain_token)

        final_reason = (
            abort_reason if abort_reason != DecodeAbortReason.NONE else _derive_abort_reason(logs)
        )
        return TADOutcome(
            token_ids=generated,
            abstained=True,
            abort_reason=final_reason,
            logs=logs,
        )


def _decode_once(
    model: ModelLike,
    oracle: Oracle,
    prefix_ids: list[int],
    cfg: TADConfig,
    rng: np.random.Generator,
    now: Callable[[], float],
    start_time: float,
) -> TADOutcome:
    V = len(model.vocab)
    generated: list[int] = []
    logs: list[TADStepLog] = []

    for t in range(cfg.max_new_tokens):
        if cfg.max_decode_ms is not None and (now() - start_time) * 1000.0 > cfg.max_decode_ms:
            logs.append(
                TADStepLog(
                    t=len(logs),
                    pi_safe=0.0,
                    picked_id=None,
                    blocked_count=0,
                    reasons_for_picked=[],
                    aborted=True,
                )
            )
            return TADOutcome(generated, True, DecodeAbortReason.TIMEOUT, logs)

        logits = model.next_logits(prefix_ids)
        if logits.shape != (V,):
            raise OracleContractError(
                f"Model returned logits of shape {logits.shape}, expected {(V,)}"
            )

        logits = logits.astype(np.float64, copy=True)
        if cfg.repetition_penalty != 1.0:
            _apply_repetition_penalty(logits, generated, cfg.repetition_penalty)

        pre_mask = np.ones(V, dtype=bool)
        if cfg.preselect_top_k > 0 or cfg.preselect_top_p < 1.0:
            probs_full = _safe_softmax(logits)
            pre_mask = _trim_by_topk_topp(probs_full, cfg.preselect_top_k, cfg.preselect_top_p)
        else:
            probs_full = _safe_softmax(logits)

        report: OracleReport = oracle.evaluate(prefix_ids, logits, model.vocab)
        safe_mask = np.array(report.safe_mask, dtype=bool, copy=True)
        if safe_mask.shape != (V,) or safe_mask.dtype != np.bool_:
            raise OracleContractError("Oracle must return bool mask of shape [V]")

        if cfg.allow_eos_escape:
            safe_mask[model.eos_id()] = True

        safe_mask &= pre_mask

        safe_mass = float(probs_full[safe_mask].sum())

        if (safe_mass < cfg.tau) or (not np.any(safe_mask)):
            logs.append(_mk_log(t, safe_mass, None, safe_mask, report, probs_full, cfg))
            return TADOutcome(generated, True, DecodeAbortReason.ABSTAIN_LOW_SAFE_MASS, logs)

        extra_mask = safe_mask.copy()
        _apply_no_repeat_ngram(prefix_ids, extra_mask, cfg.no_repeat_ngram)
        if not np.any(extra_mask):
            extra_mask = safe_mask

        next_id = _choose_id(logits, extra_mask, cfg.temperature, cfg.top_p, cfg.top_k, rng)

        if len(generated) < cfg.min_new_tokens and (
            next_id == model.eos_id() or next_id in cfg.stop_token_ids
        ):
            tmp_mask = extra_mask.copy()
            tmp_mask[next_id] = False
            if np.any(tmp_mask):
                next_id = _choose_id(logits, tmp_mask, cfg.temperature, cfg.top_p, cfg.top_k, rng)

        prefix_ids.append(next_id)
        generated.append(next_id)

        logs.append(_mk_log(t, safe_mass, next_id, extra_mask, report, probs_full, cfg))

        if next_id == model.eos_id() and len(generated) >= cfg.min_new_tokens:
            break
        if next_id in cfg.stop_token_ids and len(generated) >= cfg.min_new_tokens:
            break

    return TADOutcome(generated, False, DecodeAbortReason.NONE, logs)


def _mk_log(
    t: int,
    pi_safe: float,
    picked_id: Optional[int],
    mask: np.ndarray,
    report: OracleReport,
    probs_full: np.ndarray,
    cfg: TADConfig,
) -> TADStepLog:
    if not cfg.enable_step_logs:
        return TADStepLog(
            t=t,
            pi_safe=pi_safe,
            picked_id=picked_id,
            blocked_count=int((~mask).sum()),
            reasons_for_picked=[],
            aborted=picked_id is None,
        )
    top_k = max(1, cfg.log_top_k)
    idx = np.argsort(-probs_full)[:top_k]
    reasons = (
        sorted(report.reasons[picked_id])
        if picked_id is not None and picked_id < len(report.reasons)
        else []
    )
    return TADStepLog(
        t=t,
        pi_safe=float(pi_safe),
        picked_id=picked_id,
        blocked_count=int((~mask).sum()),
        reasons_for_picked=reasons,
        topk_ids=[int(i) for i in idx.tolist()],
        topk_probs=[float(probs_full[i]) for i in idx.tolist()],
        aborted=(picked_id is None),
    )


def _last_pi_safe(logs: list[TADStepLog]) -> float:
    for entry in reversed(logs):
        if entry.pi_safe is not None:
            return entry.pi_safe
    return 0.0


def _derive_abort_reason(logs: list[TADStepLog]) -> DecodeAbortReason:
    if not logs:
        return DecodeAbortReason.NONE
    last = logs[-1]
    if last.aborted and last.picked_id is None:
        return DecodeAbortReason.ABSTAIN_LOW_SAFE_MASS
    return DecodeAbortReason.NONE

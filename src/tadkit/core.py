"""Core primitives for truth-aware decoding."""

from __future__ import annotations

import collections.abc as cabc
import importlib.util
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml  # type: ignore[import-untyped]

if TYPE_CHECKING:  # pragma: no cover - help static type checkers
    import torch as torch_module  # type: ignore[import-not-found]
else:  # pragma: no cover - optional dependency resolved at runtime
    torch_module = None

if importlib.util.find_spec("torch") is not None:  # pragma: no cover - runtime optional dependency
    import torch as torch_module  # type: ignore[import-not-found]

torch: Any = torch_module


@dataclass(frozen=True)
class Rule:
    """Declarative constraint enforced during decoding."""

    name: str
    when_any: cabc.Sequence[str] = field(default_factory=list)
    allow_token_ids: frozenset[int] = field(default_factory=frozenset)
    abstain_on_violation: bool = False

    def allows(self, token_id: int) -> bool:
        if not self.allow_token_ids:
            return True
        return token_id in self.allow_token_ids

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "when_any": list(self.when_any),
            "allow_token_ids": sorted(self.allow_token_ids),
            "abstain_on_violation": self.abstain_on_violation,
        }


class TruthOracle:
    """Matches prompts against rules to derive decode-time constraints."""

    def __init__(self, rules: cabc.Sequence[Rule]):
        self.rules = list(rules)

    @classmethod
    def from_rules(
        cls,
        items: cabc.Iterable[dict[str, Any]],
        tokenizer: Any | None = None,
    ) -> TruthOracle:
        rules: list[Rule] = []
        for idx, item in enumerate(items):
            name = item.get("name") or f"rule_{idx}"
            when_any = tuple(item.get("when_any") or [])
            allow_ids: set[int] = set(item.get("allow_token_ids") or [])
            allow_strings: cabc.Sequence[str] = tuple(item.get("allow_strings") or [])
            if allow_strings:
                if tokenizer is None:
                    raise ValueError("allow_strings provided but tokenizer is missing")
                allow_ids.update(_strings_to_token_ids(allow_strings, tokenizer))
            rule = Rule(
                name=name,
                when_any=when_any,
                allow_token_ids=frozenset(allow_ids),
                abstain_on_violation=bool(item.get("abstain_on_violation", False)),
            )
            rules.append(rule)
        return cls(rules)

    @classmethod
    def from_payload(cls, payload: cabc.Sequence[dict[str, Any]]) -> TruthOracle:
        rules: list[Rule] = []
        for item in payload:
            allow_ids = frozenset(int(t) for t in item.get("allow_token_ids", ()))
            rule = Rule(
                name=item.get("name", "rule"),
                when_any=tuple(item.get("when_any") or []),
                allow_token_ids=allow_ids,
                abstain_on_violation=bool(item.get("abstain_on_violation", False)),
            )
            rules.append(rule)
        return cls(rules)

    @classmethod
    def from_yaml(cls, text: str, tokenizer: Any | None = None) -> TruthOracle:
        data = yaml.safe_load(text)
        items: cabc.Sequence[dict[str, Any]]
        if isinstance(data, dict) and "rules" in data:
            items = data["rules"]
        else:
            items = data or []
        if not isinstance(items, cabc.Sequence):
            raise ValueError("YAML payload must decode to a list of rules")
        return cls.from_rules(items, tokenizer=tokenizer)

    @classmethod
    def from_json(cls, text: str) -> TruthOracle:
        import json

        payload = json.loads(text)
        if isinstance(payload, dict) and "rules" in payload:
            data = payload["rules"]
        else:
            data = payload
        if not isinstance(data, cabc.Sequence):
            raise ValueError("JSON payload must decode to a list of rules")
        return cls.from_payload(data)

    def to_payload(self) -> list[dict[str, Any]]:
        return [rule.to_payload() for rule in self.rules]

    def active_for(self, prompt: str) -> list[Rule]:
        prompt_lower = prompt.lower()
        active: list[Rule] = []
        for rule in self.rules:
            if not rule.when_any:
                active.append(rule)
                continue
            for needle in rule.when_any:
                if needle.lower() in prompt_lower:
                    active.append(rule)
                    break
        return active


class TADTrace:
    """Collects decode-time events for audit and visualisation."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def log(
        self,
        *,
        step: int,
        token_id: int,
        action: str,
        rule_names: cabc.Sequence[str],
        token: str | None = None,
    ) -> None:
        self.events.append(
            {
                "step": step,
                "token_id": token_id,
                "token": token,
                "action": action,
                "rules": list(rule_names),
            }
        )

    def to_table(self, max_rows: int = 20) -> str:
        rows = self.events[:max_rows]
        header = f"{'step':>4}  {'token':<12}  {'allowed_by':<24}  {'action'}"
        lines = [header, "-" * len(header)]
        for event in rows:
            allowed = ", ".join(event["rules"]) if event["rules"] else "(n/a)"
            token = event.get("token") or str(event["token_id"])
            lines.append(f"{event['step']:>4}  {token:<12}  {allowed:<24}  {event['action']}")
        if len(self.events) > max_rows:
            lines.append(f"... {len(self.events) - max_rows} more rows")
        return "\n".join(lines)

    def to_dataframe(self):  # pragma: no cover - optional dependency
        import pandas as pd  # type: ignore[import-untyped]

        return pd.DataFrame(self.events)


class TADLogitsProcessor:
    """Hugging Face-compatible logits processor that enforces oracle rules."""

    def __init__(
        self,
        oracle: TruthOracle,
        tokenizer: Any,
        trace: TADTrace | None = None,
        abstain_token: str = "<ABSTAIN>",
    ) -> None:
        self.oracle = oracle
        self.tokenizer = tokenizer
        self.trace = trace
        self.abstain_token = abstain_token
        self._abstain_id: int | None = None

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if torch is not None and hasattr(scores, "clone"):
            return self._call_torch(input_ids, scores)
        return self._call_numpy(input_ids, scores)

    def _call_torch(self, input_ids: Any, scores: Any) -> Any:
        if torch is None:  # pragma: no cover - runtime guard
            raise RuntimeError("TADLogitsProcessor requires torch to be installed")
        if scores.ndim != 2:
            raise ValueError("scores must have shape [batch, vocab]")
        batch = scores.size(0)
        for batch_idx in range(batch):
            prompt_tokens = self._row_tokens(input_ids, batch_idx)
            prompt_text = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)
            active_rules = self.oracle.active_for(prompt_text)
            step_index = max(len(prompt_tokens) - 1, 0)
            if not active_rules:
                if self.trace:
                    token_id = int(torch.argmax(scores[batch_idx]).item())
                    self._log(step=step_index, token_id=token_id, action="pass", rules=[])
                continue

            allow_set = self._collect_allowlist(active_rules)
            original_scores = scores[batch_idx].clone()
            violation = False
            if allow_set is not None:
                allowed_idx = torch.tensor(
                    sorted(allow_set), dtype=torch.long, device=scores.device
                )
                mask = torch.full_like(scores[batch_idx], float("-inf"))
                mask[allowed_idx] = 0.0
                scores[batch_idx] = scores[batch_idx] + mask
                top_before = int(torch.argmax(original_scores).item())
                violation = top_before not in allow_set
                if violation and self.trace:
                    self._log(
                        step=step_index,
                        token_id=top_before,
                        action="block",
                        rules=[r.name for r in active_rules],
                    )
            require_abstain = any(r.abstain_on_violation for r in active_rules)
            if violation and require_abstain:
                target = self._ensure_abstain_id()
                scores[batch_idx] = float("-inf")
                scores[batch_idx, target] = 0.0
                if self.trace:
                    self._log(
                        step=step_index,
                        token_id=target,
                        action="inject",
                        rules=[r.name for r in active_rules],
                    )
                continue
            chosen = int(torch.argmax(scores[batch_idx]).item())
            self._log(
                step=step_index,
                token_id=chosen,
                action="pass",
                rules=[r.name for r in active_rules],
            )
        return scores

    def _call_numpy(self, input_ids: Any, scores: Any) -> Any:
        scores_array = np.asarray(scores, dtype=np.float64)
        if scores_array.ndim != 2:
            raise ValueError("scores must have shape [batch, vocab]")
        for batch_idx in range(scores_array.shape[0]):
            prompt_tokens = self._row_tokens(input_ids, batch_idx)
            prompt_text = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)
            active_rules = self.oracle.active_for(prompt_text)
            step_index = max(len(prompt_tokens) - 1, 0)
            if not active_rules:
                if self.trace:
                    token_id = int(np.argmax(scores_array[batch_idx]))
                    self._log(step=step_index, token_id=token_id, action="pass", rules=[])
                continue

            allow_set = self._collect_allowlist(active_rules)
            original_scores = scores_array[batch_idx].copy()
            violation = False
            if allow_set is not None:
                allowed_idx = np.array(sorted(allow_set), dtype=int)
                mask = np.full_like(scores_array[batch_idx], float("-inf"), dtype=np.float64)
                mask[allowed_idx] = 0.0
                scores_array[batch_idx] = scores_array[batch_idx] + mask
                top_before = int(np.argmax(original_scores))
                violation = top_before not in allow_set
                if violation and self.trace:
                    self._log(
                        step=step_index,
                        token_id=top_before,
                        action="block",
                        rules=[r.name for r in active_rules],
                    )
            require_abstain = any(r.abstain_on_violation for r in active_rules)
            if violation and require_abstain:
                target = self._ensure_abstain_id()
                scores_array[batch_idx] = float("-inf")
                scores_array[batch_idx, target] = 0.0
                if self.trace:
                    self._log(
                        step=step_index,
                        token_id=target,
                        action="inject",
                        rules=[r.name for r in active_rules],
                    )
                continue
            chosen = int(np.argmax(scores_array[batch_idx]))
            self._log(
                step=step_index,
                token_id=chosen,
                action="pass",
                rules=[r.name for r in active_rules],
            )
        if isinstance(scores, np.ndarray):
            scores[...] = scores_array
            return scores
        return scores_array

    @staticmethod
    def _row_tokens(input_ids: Any, batch_idx: int) -> list[int]:
        row = input_ids[batch_idx]
        if hasattr(row, "detach") and hasattr(row, "cpu"):
            raw = row.detach().cpu().tolist()
        elif hasattr(row, "tolist"):
            raw = row.tolist()
        elif isinstance(row, cabc.Iterable) and not isinstance(row, (str, bytes)):
            raw = list(row)
        else:
            raw = [row]

        def _flatten(values: Any) -> list[int]:
            if isinstance(values, cabc.Iterable) and not isinstance(values, (str, bytes)):
                items: list[int] = []
                for value in values:
                    items.extend(_flatten(value))
                return items
            return [int(values)]

        return _flatten(raw)

    def _collect_allowlist(self, rules: cabc.Sequence[Rule]) -> set[int] | None:
        allow: set[int] = set()
        for rule in rules:
            allow.update(rule.allow_token_ids)
        if not allow:
            return None
        return allow

    def _ensure_abstain_id(self) -> int:
        if self._abstain_id is not None:
            return self._abstain_id
        vocab = self.tokenizer.get_vocab()
        if self.abstain_token in vocab:
            self._abstain_id = int(vocab[self.abstain_token])
            return self._abstain_id
        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            token_id = self.tokenizer.convert_tokens_to_ids(self.abstain_token)
            if token_id != self.tokenizer.unk_token_id:
                self._abstain_id = int(token_id)
                return self._abstain_id
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is None:
            raise ValueError("tokenizer does not define an EOS or abstain token")
        self._abstain_id = int(eos)
        return self._abstain_id

    def _log(self, step: int, token_id: int, action: str, rules: cabc.Sequence[str]) -> None:
        if not self.trace:
            return
        token_text = None
        if hasattr(self.tokenizer, "convert_ids_to_tokens"):
            token_text = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        self.trace.log(
            step=step,
            token_id=token_id,
            action=action,
            rule_names=rules,
            token=token_text,
        )


def _strings_to_token_ids(strings: cabc.Sequence[str], tokenizer: Any) -> set[int]:
    token_ids: set[int] = set()
    for text in strings:
        if not text:
            continue
        pieces = tokenizer.encode(text, add_special_tokens=False)
        token_ids.update(int(tid) for tid in pieces)
    return token_ids

"""Knowledge gating via TADKit for LangChain-compatible models."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any

try:  # pragma: no cover - optional dependency
    from langchain_core.runnables import Runnable
except ModuleNotFoundError:  # pragma: no cover - fallback for environments without LangChain
    class Runnable:  # type: ignore[override]
        def invoke(self, input: Any, config: Any | None = None) -> Any:  # noqa: ANN401
            raise RuntimeError("langchain_core is required to use KnowledgeGate")

from tadkit.core import TADLogitsProcessor, TADTrace, TruthOracle


class KnowledgeGate(Runnable):
    """Wrap an LLM so each invocation receives TADKit protection."""

    def __init__(self, llm: Any, oracle: TruthOracle) -> None:
        self.llm = llm
        self.oracle = oracle
        self.last_trace: TADTrace | None = None

    def invoke(self, input: Any, config: Any | None = None) -> Any:  # noqa: D401, ANN001
        with self._install_processor() as trace:
            result = self.llm.invoke(input, config=config)
        self.last_trace = trace
        events = trace.events if trace else []
        self._attach_trace_metadata(result, events)
        return result

    @contextmanager
    def _install_processor(self):
        trace = None
        processor = None
        config_obj = None
        previous: list[Any] = []
        owner = None
        owner_attr = "logits_processor"

        model_ref = None
        tokenizer_ref = None
        if hasattr(self.llm, "model") and hasattr(self.llm, "tokenizer"):
            model_ref = getattr(self.llm, "model")
            tokenizer_ref = getattr(self.llm, "tokenizer")
        elif (
            hasattr(self.llm, "pipeline")
            and hasattr(self.llm.pipeline, "model")
            and hasattr(self.llm.pipeline, "tokenizer")
        ):
            model_ref = getattr(self.llm.pipeline, "model")
            tokenizer_ref = getattr(self.llm.pipeline, "tokenizer")

        if model_ref is not None and tokenizer_ref is not None:
            trace = TADTrace()
            processor = TADLogitsProcessor(self.oracle, tokenizer_ref, trace=trace)
            config_obj = getattr(model_ref, "generation_config", None)
            if config_obj is not None:
                owner = config_obj
                previous = list(getattr(config_obj, owner_attr, []) or [])
                setattr(config_obj, owner_attr, previous + [processor])
            elif hasattr(model_ref, owner_attr):
                owner = model_ref
                previous = list(getattr(model_ref, owner_attr, []) or [])
                setattr(model_ref, owner_attr, previous + [processor])
            elif hasattr(self.llm, owner_attr):
                owner = self.llm
                previous = list(getattr(self.llm, owner_attr, []) or [])
                setattr(self.llm, owner_attr, previous + [processor])
            else:
                processor = None
                trace = None
                owner = None
                previous = []
        try:
            yield trace
        finally:
            if processor is not None and owner is not None:
                setattr(owner, owner_attr, previous)

    def _attach_trace_metadata(self, result: Any, events: list[dict[str, Any]]) -> None:
        if hasattr(result, "response_metadata"):
            metadata = dict(getattr(result, "response_metadata") or {})
            metadata["tad_trace"] = events
            result.response_metadata = metadata
        elif isinstance(result, dict):
            metadata = dict(result.get("metadata", {}))
            metadata["tad_trace"] = events
            result["metadata"] = metadata
        else:
            setattr(result, "tad_trace", events)

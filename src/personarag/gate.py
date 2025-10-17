"""Knowledge gating via TADKit for LangChain-compatible models."""

from __future__ import annotations

import importlib
import importlib.util
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Protocol

from tadkit.core import TADLogitsProcessor, TADTrace, TruthOracle


class _RunnableProtocol(Protocol):
    def invoke(self, input: Any, config: Any | None = None) -> Any: ...


if TYPE_CHECKING:
    RunnableProtocol = _RunnableProtocol
else:
    RunnableProtocol = None

_LANGCHAIN_BASE_SPEC = importlib.util.find_spec("langchain_core")
if _LANGCHAIN_BASE_SPEC is not None:
    _RUNNABLE_SPEC = importlib.util.find_spec("langchain_core.runnables")
else:
    _RUNNABLE_SPEC = None

if _RUNNABLE_SPEC is not None and not TYPE_CHECKING:
    langchain_runnables = importlib.import_module("langchain_core.runnables")
    RunnableProtocol = langchain_runnables.Runnable


class _FallbackRunnable:
    def invoke(self, input: Any, config: Any | None = None) -> Any:  # noqa: D401, ANN401
        raise RuntimeError("langchain_core is required to use KnowledgeGate")


Runnable: type[Any]
if RunnableProtocol is not None:
    Runnable = RunnableProtocol  # type: ignore[assignment]
else:
    Runnable = _FallbackRunnable


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
        trace: TADTrace | None = None
        processor: TADLogitsProcessor | None = None
        config_obj = None
        owner: Any = None
        sentinel = object()
        previous: Any = sentinel

        model_ref = None
        tokenizer_ref = None
        if hasattr(self.llm, "model") and hasattr(self.llm, "tokenizer"):
            model_ref = self.llm.model
            tokenizer_ref = self.llm.tokenizer
        elif (
            hasattr(self.llm, "pipeline")
            and hasattr(self.llm.pipeline, "model")
            and hasattr(self.llm.pipeline, "tokenizer")
        ):
            pipeline = self.llm.pipeline
            model_ref = pipeline.model
            tokenizer_ref = pipeline.tokenizer

        if model_ref is not None and tokenizer_ref is not None:
            trace = TADTrace()
            processor = TADLogitsProcessor(self.oracle, tokenizer_ref, trace=trace)
            has_generation_config = hasattr(model_ref, "generation_config")
            config_obj = model_ref.generation_config if has_generation_config else None
            if config_obj is not None:
                owner = config_obj
                if hasattr(config_obj, "logits_processor"):
                    previous = config_obj.logits_processor
                    existing = list(previous or [])
                else:
                    previous = sentinel
                    existing = []
                existing.append(processor)
                config_obj.logits_processor = existing
            elif hasattr(model_ref, "logits_processor"):
                owner = model_ref
                previous = model_ref.logits_processor
                existing = list(previous or [])
                existing.append(processor)
                model_ref.logits_processor = existing
            elif hasattr(self.llm, "logits_processor"):
                owner = self.llm
                previous = self.llm.logits_processor
                existing = list(previous or [])
                existing.append(processor)
                self.llm.logits_processor = existing
            else:
                processor = None
                trace = None
                owner = None
                previous = None
        try:
            yield trace
        finally:
            if processor is not None and owner is not None:
                if previous is sentinel:
                    if hasattr(owner, "logits_processor"):
                        del owner.logits_processor
                elif previous is None:
                    owner.logits_processor = None
                else:
                    owner.logits_processor = previous

    def _attach_trace_metadata(self, result: Any, events: list[dict[str, Any]]) -> None:
        if hasattr(result, "response_metadata"):
            metadata = dict(result.response_metadata or {})
            metadata["tad_trace"] = events
            result.response_metadata = metadata
        elif isinstance(result, dict):
            metadata = dict(result.get("metadata", {}))
            metadata["tad_trace"] = events
            result["metadata"] = metadata
        else:
            result.tad_trace = events

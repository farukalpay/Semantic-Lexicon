"""Typer-based CLI entry point for TADKit."""

from __future__ import annotations

import csv
import json
import pathlib
from typing import Optional

import typer

from .core import TADLogitsProcessor, TADTrace, TruthOracle

app = typer.Typer(help="Truth-Aware Decoding utilities")

SOURCE_ARGUMENT = typer.Argument(..., help="CSV or JSON file with rule definitions")
OUT_OPTION = typer.Option(..., "--out", help="Output JSON file")
TOKENIZER_OPTION = typer.Option(None, help="Tokenizer identifier for allow_strings")

ORACLE_OPTION = typer.Option(..., "--oracle", help="Compiled oracle JSON")
MODEL_OPTION = typer.Option("sshleifer/tiny-gpt2", help="HF causal LM identifier")
PROMPT_OPTION = typer.Option("Q: What is the capital of France?\nA:", help="Prompt to decode")
MAX_TOKENS_OPTION = typer.Option(20, help="Number of tokens to generate")


def _load_tokenizer(tokenizer_id: str | None):
    if not tokenizer_id:
        return None
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    return AutoTokenizer.from_pretrained(tokenizer_id)


@app.command()
def compile(
    source: pathlib.Path = SOURCE_ARGUMENT,
    out: pathlib.Path = OUT_OPTION,
    tokenizer: Optional[str] = TOKENIZER_OPTION,
) -> None:
    """Compile rules from CSV/JSON into a JSON oracle payload."""

    source_path = pathlib.Path(source)
    out_path = pathlib.Path(out)

    if source_path.suffix.lower() == ".csv":
        rules = _load_rules_from_csv(source_path)
    else:
        with source_path.open("r", encoding="utf8") as handle:
            data = json.load(handle)
        rules = data["rules"] if isinstance(data, dict) and "rules" in data else data
    tok = _load_tokenizer(tokenizer)
    oracle = TruthOracle.from_rules(rules, tokenizer=tok)
    payload = {"rules": oracle.to_payload()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf8") as handle:
        json.dump(payload, handle, indent=2)
    typer.echo(f"Wrote {len(payload['rules'])} rules to {out_path}")


def _load_rules_from_csv(path: pathlib.Path) -> list[dict[str, object]]:
    rules: list[dict[str, object]] = []
    csv_path = pathlib.Path(path)
    with csv_path.open("r", encoding="utf8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            when = [frag.strip() for frag in (row.get("when") or "").split("|") if frag.strip()]
            allow = [frag.strip() for frag in (row.get("allow") or "").split("|") if frag.strip()]
            abstain = str(row.get("abstain", "0")).strip().lower() in {"1", "true", "yes"}
            rules.append(
                {
                    "name": row.get("name") or f"rule_{idx}",
                    "when_any": when,
                    "allow_strings": allow,
                    "abstain_on_violation": abstain,
                }
            )
    return rules


@app.command()
def demo(
    oracle: pathlib.Path = ORACLE_OPTION,
    model: str = MODEL_OPTION,
    prompt: str = PROMPT_OPTION,
    max_new_tokens: int = MAX_TOKENS_OPTION,
) -> None:
    """Run a small decoding demo against a Hugging Face model."""

    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCausalLM,
        AutoTokenizer,
        LogitsProcessorList,
    )

    oracle_path = pathlib.Path(oracle)
    with oracle_path.open("r", encoding="utf8") as handle:
        payload = json.load(handle)
    truth = TruthOracle.from_payload(payload["rules"])
    tokenizer = AutoTokenizer.from_pretrained(model)
    model_ref = AutoModelForCausalLM.from_pretrained(model)
    trace = TADTrace()
    processor = TADLogitsProcessor(truth, tokenizer, trace=trace)
    processors = LogitsProcessorList([processor])
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model_ref.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        logits_processor=processors,
        do_sample=False,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    typer.echo(text)
    typer.echo("--- TAD Trace ---")
    typer.echo(trace.to_table())


__all__ = ["app"]

"""Typer-based CLI entry point for TADKit."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import typer

from .core import TADLogitsProcessor, TADTrace, TruthOracle

app = typer.Typer(help="Truth-Aware Decoding utilities")


def _load_tokenizer(tokenizer_id: str | None):
    if not tokenizer_id:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_id)


@app.command()
def compile(
    source: Path = typer.Argument(..., help="CSV or JSON file with rule definitions"),
    out: Path = typer.Option(..., "--out", help="Output JSON file"),
    tokenizer: Optional[str] = typer.Option(None, help="Tokenizer identifier for allow_strings"),
) -> None:
    """Compile rules from CSV/JSON into a JSON oracle payload."""

    if source.suffix.lower() == ".csv":
        rules = _load_rules_from_csv(source)
    else:
        with source.open("r", encoding="utf8") as handle:
            data = json.load(handle)
        rules = data["rules"] if isinstance(data, dict) and "rules" in data else data
    tok = _load_tokenizer(tokenizer)
    oracle = TruthOracle.from_rules(rules, tokenizer=tok)
    payload = {"rules": oracle.to_payload()}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf8") as handle:
        json.dump(payload, handle, indent=2)
    typer.echo(f"Wrote {len(payload['rules'])} rules to {out}")


def _load_rules_from_csv(path: Path) -> list[dict[str, object]]:
    rules: list[dict[str, object]] = []
    with path.open("r", encoding="utf8") as handle:
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
    oracle: Path = typer.Option(..., "--oracle", help="Compiled oracle JSON"),
    model: str = typer.Option("sshleifer/tiny-gpt2", help="HF causal LM identifier"),
    prompt: str = typer.Option("Q: What is the capital of France?\nA:", help="Prompt to decode"),
    max_new_tokens: int = typer.Option(20, help="Number of tokens to generate"),
) -> None:
    """Run a small decoding demo against a Hugging Face model."""

    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

    with oracle.open("r", encoding="utf8") as handle:
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

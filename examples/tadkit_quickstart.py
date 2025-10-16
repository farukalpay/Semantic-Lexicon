"""Quickstart example for TADKit."""
from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from tadkit import TADLogitsProcessor, TADTrace, TruthOracle


def main() -> None:
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    oracle = TruthOracle.from_rules(
        [
            {
                "name": "country_capitals",
                "when_any": ["capital of France", "France capital"],
                "allow_strings": [" Paris"],
                "abstain_on_violation": True,
            }
        ],
        tokenizer=tokenizer,
    )

    trace = TADTrace()
    processor = LogitsProcessorList([TADLogitsProcessor(oracle, tokenizer, trace=trace)])

    prompt = "Q: What is the capital of France?\nA:"
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_new_tokens=12,
        logits_processor=processor,
        do_sample=False,
    )

    print(tokenizer.decode(output[0], skip_special_tokens=True))
    print("--- TAD Trace (compact) ---")
    print(trace.to_table(max_rows=6))


if __name__ == "__main__":
    main()

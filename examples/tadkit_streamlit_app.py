"""Streamlit demo for inspecting truth-aware decoding traces."""
from __future__ import annotations

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from tadkit import TADLogitsProcessor, TADTrace, TruthOracle


st.set_page_config(page_title="TADKit — Truth-Aware Decoding")
st.title("TADKit — Truth-Aware Decoding")

prompt = st.text_area("Prompt", "Q: What is the capital of France?\nA:")
rules_text = st.text_area(
    "Rules (YAML)",
    """
- name: country_capitals
  when_any: ["capital of France", "France capital"]
  allow_strings: [" Paris"]
  abstain_on_violation: true
""".strip(),
)

model_id = st.text_input("Model", "gpt2")

if st.button("Generate"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    oracle = TruthOracle.from_yaml(rules_text, tokenizer=tokenizer)
    trace = TADTrace()
    processor = LogitsProcessorList([TADLogitsProcessor(oracle, tokenizer, trace=trace)])
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_new_tokens=20,
        logits_processor=processor,
        do_sample=False,
    )
    st.write("**Output**:", tokenizer.decode(output[0], skip_special_tokens=True))
    st.write("**Trace**")
    st.text(trace.to_table())

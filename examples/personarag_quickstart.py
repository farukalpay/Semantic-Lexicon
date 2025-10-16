"""Quickstart example for PersonaRAG."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline

from personarag import BrandStyle, KnowledgeGate, PersonaPolicyEXP3
from tadkit import TruthOracle


def main() -> None:
    personas = [
        BrandStyle(
            name="WarmGuide",
            system="You are warm, encouraging, and concise. Prefer lists of 3.",
            signature="—Team WarmGuide",
        ),
        BrandStyle(
            name="CrispPro",
            system="You are direct, technical, and to-the-point. Use short sentences.",
            signature="—Crisp",
        ),
    ]
    policy = PersonaPolicyEXP3(personas)

    oracle = TruthOracle.from_rules(
        [
            {
                "name": "capitals",
                "when_any": ["capital of France"],
                "allow_strings": [" Paris"],
                "abstain_on_violation": True,
            }
        ]
    )

    def route_persona(inputs):
        choice = policy.choose(context=inputs["question"])
        return {"persona": choice}

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "{persona.system}\nUse only the provided CONTEXT for facts. If missing, abstain.",
            ),
            (
                "human",
                "QUESTION: {question}\nCONTEXT:\n{context}\n\nAnswer in brand voice and cite sources.",
            ),
        ]
    )

    model_id = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipeline = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=64,
        do_sample=False,
    )
    llm = HuggingFacePipeline(pipeline=pipeline)
    gated_llm = KnowledgeGate(llm, oracle=oracle)

    # Hosted-models note: `KnowledgeGate` can wrap providers like OpenAI via
    # `ChatOpenAI`, but it only attaches trace metadata when decode-time logits
    # processors are unavailable. Example:
    # from langchain_openai import ChatOpenAI
    # hosted_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # hosted_gate = KnowledgeGate(hosted_llm, oracle=oracle)  # trace-only fallback

    chain = (
        RunnableParallel(
            question=RunnablePassthrough(),
            route=RunnableLambda(route_persona),
            context=RunnableLambda(lambda q: ["Paris is the capital of France."]),
        )
        | RunnableLambda(
            lambda d: {
                "question": d["question"],
                "context": "\n".join(d["context"]),
                "persona": d["route"]["persona"],
            }
        )
        | prompt
        | gated_llm
    )

    response = chain.invoke("What is the capital of France?")
    print(getattr(response, "content", response))
    policy.update(reward=1.0)
    print("Updated persona weights:", policy.telemetry())


if __name__ == "__main__":
    main()

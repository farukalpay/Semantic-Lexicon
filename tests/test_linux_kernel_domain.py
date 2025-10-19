"""Stress tests for Linux and Ubuntu kernel intent and knowledge coverage."""

from __future__ import annotations

from collections.abc import Iterable

from semantic_lexicon.intent import IntentClassifier, IntentExample
from semantic_lexicon.knowledge import KnowledgeEdge, KnowledgeNetwork
from semantic_lexicon.utils import read_jsonl


def _load_training_examples() -> list[IntentExample]:
    examples: list[IntentExample] = []
    for record in read_jsonl("src/semantic_lexicon/data/intent.jsonl"):
        examples.append(
            IntentExample(
                text=str(record["text"]),
                intent=str(record["intent"]),
                feedback=0.92,
            )
        )
    return examples


def _load_kernel_validation_records() -> list[tuple[str, str, float]]:
    prompts: list[tuple[str, str, float]] = []
    for record in read_jsonl("src/semantic_lexicon/data/cross_domain_validation.jsonl"):
        if str(record.get("domain")) != "linux_kernel_operations":
            continue
        text = str(record["text"])
        intent = str(record["intent"])
        feedback = float(record.get("feedback", 0.9))
        prompts.append((text, intent, feedback))
    return prompts


def _load_kernel_edges() -> Iterable[KnowledgeEdge]:
    for record in read_jsonl("src/semantic_lexicon/data/knowledge.jsonl"):
        head = str(record["head"])
        tail = str(record["tail"])
        if not (
            head.startswith("ubuntu_kernel")
            or head.startswith("linux_kernel")
            or tail.startswith("ubuntu_kernel")
            or tail.startswith("linux_kernel")
        ):
            continue
        yield KnowledgeEdge(
            head=head,
            relation=str(record["relation"]),
            tail=tail,
        )


def test_linux_kernel_validation_prompts_are_classified() -> None:
    training = _load_training_examples()
    prompts = _load_kernel_validation_records()
    assert prompts, "Expected linux_kernel_operations prompts in validation set"
    classifier = IntentClassifier()
    classifier.fit(training)
    for text, intent, feedback in prompts:
        probabilities = classifier.predict_proba(text)
        predicted = max(probabilities.items(), key=lambda item: item[1])[0]
        assert predicted == intent
        reward = float(classifier.reward(text, predicted, intent, feedback))
        assert reward >= 0.74


def test_linux_kernel_knowledge_neighbours_cover_layers() -> None:
    edges = list(_load_kernel_edges())
    assert any(edge.head == "ubuntu_kernel_stack" for edge in edges)
    network = KnowledgeNetwork()
    network.fit(edges)
    stack_neighbours = network.neighbours("ubuntu_kernel_stack", top_k=40)
    neighbour_names = {name for name, _ in stack_neighbours}
    assert "livepatch_service" in neighbour_names
    assert "kernel_oops_reports" in neighbour_names
    assert "canonical_crash_dump_portal" in neighbour_names
    assert "ftrace_runtime_tracing" in neighbour_names
    observability_neighbours = network.neighbours("linux_kernel_observability_suite", top_k=5)
    observability_names = {name for name, _ in observability_neighbours}
    assert "ubuntu_kernel_stack" in observability_names

# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

from __future__ import annotations

from semantic_lexicon.knowledge import KnowledgeEdge, KnowledgeNetwork


def test_knowledge_neighbours() -> None:
    network = KnowledgeNetwork()
    edges = [
        KnowledgeEdge(head="ai", relation="related_to", tail="ml"),
        KnowledgeEdge(head="ai", relation="related_to", tail="nlp"),
        KnowledgeEdge(head="ml", relation="related_to", tail="data"),
    ]
    network.fit(edges)
    neighbours = network.neighbours("ai", top_k=2)
    assert len(neighbours) == 2
    assert {name for name, _ in neighbours} <= {"ml", "nlp", "data"}

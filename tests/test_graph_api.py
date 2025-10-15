# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from semantic_lexicon.graph_memory import InMemoryGraph


def test_roundtrip_entities_and_facts(tmp_path) -> None:
    graph = InMemoryGraph()
    paris = graph.upsert_entity("Paris", aliases=["City of Paris"])
    france = graph.upsert_entity("France")
    graph.upsert_relation(paris, "capital_of", france)
    path = tmp_path / "graph.json"
    graph.save(str(path))

    loaded = InMemoryGraph()
    loaded.load(str(path))

    assert loaded.find_entity_by_surface("paris") == paris
    assert france in loaded.objects(paris, "capital_of")
    assert loaded.has_fact(paris, "capital_of", france)

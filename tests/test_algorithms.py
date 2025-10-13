# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

import numpy as np
import pytest

from semantic_lexicon.algorithms import (
    EXP3,
    AnytimeEXP3,
    EXP3Config,
    TopicPureRetrievalConfig,
    TopicPureRetriever,
)

TopicDataset = dict[str, np.ndarray | list[str]]


def test_exp3_initialises_uniform_probabilities() -> None:
    config = EXP3Config(num_arms=2, horizon=8, rng=np.random.default_rng(0))
    solver = EXP3(config)
    assert np.allclose(solver.probabilities, np.full(2, 0.5))


def test_exp3_weight_update_boosts_selected_arm() -> None:
    rng = np.random.default_rng(1)
    config = EXP3Config(num_arms=3, horizon=5, rng=rng)
    solver = EXP3(config)
    weights_before = solver.weights
    arm = solver.select_arm()
    solver.update(1.0)
    weights_after = solver.weights
    assert weights_after[arm] > weights_before[arm]
    untouched_indices = [i for i in range(3) if i != arm]
    assert np.allclose(weights_after[untouched_indices], weights_before[untouched_indices])


def test_exp3_rejects_rewards_outside_unit_interval() -> None:
    config = EXP3Config(num_arms=2, horizon=3, rng=np.random.default_rng(2))
    solver = EXP3(config)
    solver.select_arm()
    with pytest.raises(ValueError):
        solver.update(1.5)


def test_exp3_enforces_horizon_limit() -> None:
    config = EXP3Config(num_arms=2, horizon=1, rng=np.random.default_rng(3))
    solver = EXP3(config)
    solver.select_arm()
    solver.update(0.5)
    with pytest.raises(RuntimeError):
        solver.select_arm()


def test_anytime_exp3_doubles_epoch_horizon() -> None:
    rng = np.random.default_rng(4)
    agent = AnytimeEXP3(num_arms=2, rng=rng)
    horizons = []
    for _ in range(4):
        horizons.append(agent.epoch_horizon)
        agent.select_arm()
        agent.update(0.0)
    assert horizons == [1, 2, 2, 4]


def test_anytime_exp3_resets_distribution_between_epochs() -> None:
    rng = np.random.default_rng(5)
    agent = AnytimeEXP3(num_arms=2, rng=rng)
    agent.select_arm()
    agent.update(1.0)
    assert np.allclose(agent.probabilities, np.full(2, 0.5))


@pytest.fixture
def topic_dataset() -> TopicDataset:
    concept_ids = [f"c{i}" for i in range(6)]
    query_ids = [f"q{i}" for i in range(4)]
    concept_embeddings = np.array(
        [
            [1.0, 0.1, 0.0],
            [0.95, -0.05, 0.05],
            [1.05, 0.05, -0.02],
            [0.0, 1.0, 0.1],
            [0.05, 0.95, -0.05],
            [-0.05, 1.1, 0.0],
        ],
        dtype=float,
    )
    query_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.05],
            [0.0, 1.0, 0.0],
            [0.1, 0.9, -0.02],
        ],
        dtype=float,
    )
    concept_labels = np.array(["A", "A", "A", "B", "B", "B"], dtype=object)
    query_labels = np.array(["A", "A", "B", "B"], dtype=object)
    return {
        "concept_ids": concept_ids,
        "concept_embeddings": concept_embeddings,
        "query_ids": query_ids,
        "query_embeddings": query_embeddings,
        "concept_labels": concept_labels,
        "query_labels": query_labels,
    }


def _train_topic_retriever(dataset: TopicDataset) -> TopicPureRetriever:
    config = TopicPureRetrievalConfig(
        k=2,
        margin=0.4,
        lambda_reg=5e-3,
        beta_reg=1e-3,
        learning_rate=0.1,
        epochs=80,
        negative_samples=2,
        random_state=0,
    )
    retriever = TopicPureRetriever(config)
    retriever.fit(
        dataset["concept_ids"],
        dataset["concept_embeddings"],
        dataset["query_ids"],
        dataset["query_embeddings"],
        concept_labels=dataset["concept_labels"],
        query_labels=dataset["query_labels"],
    )
    return retriever


def test_topic_pure_retriever_aligns_hits_with_topic(topic_dataset: TopicDataset) -> None:
    retriever = _train_topic_retriever(topic_dataset)
    concept_labels = topic_dataset["concept_labels"]
    query_labels = topic_dataset["query_labels"]
    for query_id, expected_label in zip(topic_dataset["query_ids"], query_labels):
        top = retriever.top_k_for_query_id(query_id, k=1)
        assert top, "retriever should return at least one concept"
        top_concept, _ = top[0]
        label = concept_labels[retriever._concept_index[top_concept]]
        assert label == expected_label
    assert retriever.triplet_violation_rate < 0.2


def test_topic_pure_retriever_normalises_embeddings(topic_dataset: TopicDataset) -> None:
    retriever = _train_topic_retriever(topic_dataset)
    concept_norms = np.linalg.norm(retriever.concept_embeddings_, axis=1)
    query_norms = np.linalg.norm(retriever.query_embeddings_, axis=1)
    assert np.allclose(concept_norms, np.ones_like(concept_norms), atol=1e-6)
    assert np.allclose(query_norms, np.ones_like(query_norms), atol=1e-6)
    eigenvalues = np.linalg.eigvalsh(retriever.M_)
    assert np.all(eigenvalues >= -1e-8)
    assert np.all((retriever.gate_ >= -1e-8) & (retriever.gate_ <= 1.0 + 1e-8))


def test_topic_pure_retriever_reports_high_purity(topic_dataset: TopicDataset) -> None:
    retriever = _train_topic_retriever(topic_dataset)
    for query_id in topic_dataset["query_ids"]:
        purity = retriever.purity_at_k(query_id, k=2)
        assert purity == pytest.approx(1.0)
    assert retriever.gate_sparsity <= 1.0

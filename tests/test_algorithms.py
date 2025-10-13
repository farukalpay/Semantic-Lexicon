# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

from typing import TypedDict

import numpy as np
import pytest
from numpy.typing import NDArray

from semantic_lexicon.algorithms import (
    EXP3,
    AnytimeEXP3,
    EXP3Config,
    TopicPureRetrievalConfig,
    TopicPureRetriever,
)
from semantic_lexicon.algorithms.topic_pure_retrieval import _compute_covariance, _project_to_psd


class TopicDataset(TypedDict):
    concept_ids: list[str]
    concept_embeddings: NDArray[np.float64]
    query_ids: list[str]
    query_embeddings: NDArray[np.float64]
    concept_labels: NDArray[np.str_]
    query_labels: NDArray[np.str_]


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
        dtype=np.float64,
    )
    query_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.05],
            [0.0, 1.0, 0.0],
            [0.1, 0.9, -0.02],
        ],
        dtype=np.float64,
    )
    concept_labels = np.array(["A", "A", "A", "B", "B", "B"], dtype=np.str_)
    query_labels = np.array(["A", "A", "B", "B"], dtype=np.str_)
    return TopicDataset(
        concept_ids=concept_ids,
        concept_embeddings=concept_embeddings,
        query_ids=query_ids,
        query_embeddings=query_embeddings,
        concept_labels=concept_labels,
        query_labels=query_labels,
    )


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
        concept_labels=list(dataset["concept_labels"].tolist()),
        query_labels=list(dataset["query_labels"].tolist()),
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
    assert retriever.concept_embeddings_ is not None
    assert retriever.query_embeddings_ is not None
    concept_norms = np.linalg.norm(retriever.concept_embeddings_, axis=1)
    query_norms = np.linalg.norm(retriever.query_embeddings_, axis=1)
    assert np.allclose(concept_norms, np.ones_like(concept_norms), atol=1e-6)
    assert np.allclose(query_norms, np.ones_like(query_norms), atol=1e-6)
    assert retriever.M_ is not None
    eigenvalues = np.linalg.eigvalsh(retriever.M_)
    assert np.all(eigenvalues >= -1e-8)
    assert retriever.gate_ is not None
    assert np.all((retriever.gate_ >= -1e-8) & (retriever.gate_ <= 1.0 + 1e-8))


def test_topic_pure_retriever_reports_high_purity(topic_dataset: TopicDataset) -> None:
    retriever = _train_topic_retriever(topic_dataset)
    for query_id in topic_dataset["query_ids"]:
        purity = retriever.purity_at_k(query_id, k=2)
        assert purity == pytest.approx(1.0)
    assert retriever.gate_sparsity <= 1.0


def _summarise_retrieval(
    retriever: TopicPureRetriever,
    dataset: TopicDataset,
    k: int,
) -> tuple[list[tuple[str, ...]], dict[str, float], float]:
    topk_sets: list[tuple[str, ...]] = []
    purities: dict[str, float] = {}
    for query_id in dataset["query_ids"]:
        topk = retriever.top_k_for_query_id(query_id, k=k)
        topk_sets.append(tuple(concept_id for concept_id, _ in topk))
        purities[query_id] = retriever.purity_at_k(query_id, k=k)
    return topk_sets, purities, retriever.triplet_violation_rate


def test_topic_pure_retriever_is_deterministic(topic_dataset: TopicDataset) -> None:
    retriever_first = _train_topic_retriever(topic_dataset)
    retriever_second = _train_topic_retriever(topic_dataset)

    topk_first, purities_first, violation_first = _summarise_retrieval(
        retriever_first, topic_dataset, k=2
    )
    topk_second, purities_second, violation_second = _summarise_retrieval(
        retriever_second, topic_dataset, k=2
    )

    assert topk_first == topk_second
    assert purities_first == purities_second
    assert violation_first == pytest.approx(violation_second, abs=0.0)


def test_topic_pure_retriever_whitening_is_isotropic(topic_dataset: TopicDataset) -> None:
    retriever = _train_topic_retriever(topic_dataset)
    assert retriever.whitened_concepts_ is not None
    concept_whitened = retriever.whitened_concepts_
    assert concept_whitened.dtype == np.float64
    concept_mean = np.mean(concept_whitened, axis=0)
    assert np.allclose(concept_mean, np.zeros_like(concept_mean), atol=1e-6)
    concept_cov = _compute_covariance(concept_whitened)
    assert concept_cov.dtype == np.float64
    assert np.allclose(concept_cov, np.eye(concept_cov.shape[0]), atol=1e-5)

    assert retriever.whitened_queries_ is not None
    query_whitened = retriever.whitened_queries_
    assert query_whitened.dtype == np.float64
    query_mean = np.mean(query_whitened, axis=0)
    assert np.allclose(query_mean, np.zeros_like(query_mean), atol=1e-6)
    query_cov = _compute_covariance(query_whitened)
    assert query_cov.dtype == np.float64
    assert np.isfinite(query_cov).all()
    assert np.linalg.norm(query_cov - np.eye(query_cov.shape[0])) < 1.5


def test_topic_pure_retriever_psd_and_label_constraints(topic_dataset: TopicDataset) -> None:
    retriever = _train_topic_retriever(topic_dataset)
    assert retriever.M_ is not None
    M = retriever.M_
    assert M.dtype == np.float64
    assert np.allclose(M, M.T, atol=1e-8)
    eigenvalues = np.linalg.eigvalsh(M)
    assert eigenvalues.min() >= -1e-8
    projected = _project_to_psd(M)
    assert np.allclose(projected, M, atol=1e-8)
    assert np.allclose(projected, _project_to_psd(projected), atol=1e-8)

    assert retriever.gate_ is not None
    gate = retriever.gate_
    assert gate.dtype == np.float64
    assert np.all((gate >= -1e-10) & (gate <= 1.0 + 1e-10))

    assert retriever.concept_labels_ is not None
    concept_labels = retriever.concept_labels_
    assert concept_labels.dtype == np.int64
    num_topics = int(concept_labels.max()) + 1
    assert np.array_equal(np.unique(concept_labels), np.arange(num_topics))

    assert retriever.query_labels_ is not None
    query_labels = retriever.query_labels_
    assert query_labels.dtype == np.int64
    assert np.all((query_labels >= 0) & (query_labels < num_topics))

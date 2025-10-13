# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""Persona-aware response generation."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Optional, cast

import numpy as np
from numpy.typing import NDArray

from .config import GeneratorConfig
from .embeddings import GloVeEmbeddings
from .knowledge import KnowledgeNetwork
from .logging import configure_logging
from .persona import PersonaProfile
from .template_learning import BalancedTutorPredictor
from .templates import render_balanced_tutor_response
from .utils import tokenize

LOGGER = configure_logging(logger_name=__name__)


@dataclass
class GenerationResult:
    response: str
    intents: list[str]
    knowledge_hits: list[str]
    phrases: list[str] = field(default_factory=list)


@dataclass
class PhraseGuidance:
    text: str
    phrases: list[str]


@dataclass
class PhraseCandidate:
    tokens: tuple[str, ...]
    lemmas: tuple[str, ...]
    text: str
    embedding: NDArray[np.float64]
    relevance: float
    tfidf: float
    bonus: float
    score: float
    ngrams: set[tuple[str, ...]]


ALPHA = 0.6
BETA = 0.3
GAMMA = 0.1
MMR_LAMBDA = 0.7
MMR_ETA = 1.0
OVERLAP_MU = 0.3
PMI_BONUS_CAP = 2.0
PHRASE_LIMIT = 3
LENGTH_BONUS = 0.05

PHRASE_EXPANSIONS = {
    "public speaking": ["practice routine", "feedback loops"],
    "matrix multiplication": ["linear transformations", "dot products"],
    "machine learning": ["supervised learning", "generalization error"],
}

DEFAULT_FALLBACK_TOPICS = ["Key Insight", "Next Step", "Guiding Question"]

ACTIONS_BY_INTENT = {
    "how_to": ["Explore", "Practice", "Reflect"],
    "definition": ["Define", "Explore", "Compare"],
}

VERB_BLACKLIST = {
    "explain",
    "improve",
    "define",
    "describe",
    "outline",
    "what",
    "how",
}


class PersonaGenerator:
    """Sample-based generator conditioned on persona vector."""

    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        embeddings: Optional[GloVeEmbeddings] = None,
        knowledge: Optional[KnowledgeNetwork] = None,
        template_predictor: Optional[BalancedTutorPredictor] = None,
    ) -> None:
        self.config = config or GeneratorConfig()
        self.embeddings = embeddings
        self.knowledge = knowledge
        self.template_predictor = template_predictor

    def generate(
        self,
        prompt: str,
        persona: PersonaProfile,
        intents: Iterable[str],
    ) -> GenerationResult:
        tokens = tokenize(prompt)
        vectors = self.embeddings.encode_tokens(tokens) if self.embeddings else np.zeros((0,))
        if vectors.size:
            prompt_vector = vectors.mean(axis=0)
        else:
            prompt_vector = np.zeros((persona.vector.size,), dtype=float)
        persona_vector = _match_dimensions(persona.vector, prompt_vector)
        semantic_vector = 0.6 * prompt_vector + 0.4 * persona_vector
        intents_list = list(intents)
        primary_intent = next(iter(intents_list), "general")
        phrase_guidance = _build_phrase_guidance(
            tokens,
            semantic_vector,
            self.embeddings,
            self.knowledge,
        )
        topics: list[str] = []
        actions: list[str] = []
        predicted_intent: Optional[str] = None
        if self.template_predictor is not None:
            prediction = self.template_predictor.predict_variables(prompt)
            predicted_intent = prediction.intent
            topics = list(prediction.topics)
            actions = list(prediction.actions)
        if predicted_intent:
            primary_intent = predicted_intent
            if predicted_intent not in intents_list:
                intents_list.insert(0, predicted_intent)
        if not topics:
            topics = _ensure_topics(tokens, phrase_guidance.phrases)
            actions = _actions_for_intent(primary_intent, len(topics))
            topics = topics[: len(actions)]
        else:
            limit = min(len(topics), len(actions)) if actions else 0
            if not limit:
                actions = _actions_for_intent(primary_intent, len(topics))
                limit = min(len(topics), len(actions))
            topics = topics[:limit]
            actions = actions[:limit]
        if topics:
            base_line = render_balanced_tutor_response(
                prompt=prompt,
                intent=primary_intent,
                topics=topics,
                actions=actions,
            )
        else:
            base_line = _build_intro(prompt, primary_intent)
        related, hits = _build_related_topics(
            self.knowledge,
            topics,
            tokens,
        )
        response_parts = [segment for segment in [base_line, related] if segment]
        if not response_parts:
            response_parts.append(
                "I'm here to help, but I need a bit more detail to respond meaningfully."
            )
        response = " ".join(response_parts)
        return GenerationResult(
            response=response,
            intents=intents_list,
            knowledge_hits=hits,
            phrases=topics,
        )


def _match_dimensions(persona_vector: np.ndarray, prompt_vector: np.ndarray) -> np.ndarray:
    """Pad or truncate persona vector to match prompt dimensionality."""

    if persona_vector.size == prompt_vector.size:
        return persona_vector
    if persona_vector.size > prompt_vector.size:
        return persona_vector[: prompt_vector.size]
    padded = np.zeros_like(prompt_vector)
    padded[: persona_vector.size] = persona_vector
    return padded


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "do",
    "for",
    "from",
    "how",
    "i",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "to",
    "what",
    "with",
    "you",
    "my",
}


def _normalise_token(token: str) -> str:
    return "".join(char for char in token.lower() if char.isalpha())


def _build_intro(prompt: str, intent: str) -> str:
    prompt_text = prompt.strip()
    if not prompt_text:
        prompt_text = "this topic"
    if not prompt_text.endswith((".", "!", "?")):
        prompt_text = f"{prompt_text}."
    intent_label = intent or "general"
    return (
        f"From a balanced tutor perspective, let's look at {prompt_text} "
        f"This ties closely to the '{intent_label}' intent I detected."
    )


def _build_phrase_guidance(
    tokens: Sequence[str],
    prompt_vector: np.ndarray,
    embeddings: Optional[GloVeEmbeddings],
    knowledge: Optional[KnowledgeNetwork],
) -> PhraseGuidance:
    normalised_tokens = [_normalise_token(token) for token in tokens]
    normalised_tokens = [token for token in normalised_tokens if token]
    if not normalised_tokens:
        return PhraseGuidance(text="", phrases=[])
    candidates = _enumerate_phrase_candidates(
        normalised_tokens,
        prompt_vector,
        embeddings,
        knowledge,
    )
    selected = _select_phrases(candidates, normalised_tokens)
    phrases = [_format_phrase(candidate.tokens) for candidate in selected]
    return PhraseGuidance(text="", phrases=phrases)


def _ensure_topics(tokens: Sequence[str], phrases: Sequence[str]) -> list[str]:
    topics = list(phrases[:PHRASE_LIMIT])
    target_count = min(PHRASE_LIMIT, 3)
    if len(topics) >= target_count:
        return topics[:target_count]
    needed = target_count - len(topics)
    topics.extend(_fallback_topics(tokens, needed, existing=topics))
    return topics


def _fallback_topics(tokens: Sequence[str], needed: int, existing: Sequence[str]) -> list[str]:
    seen = {topic.lower() for topic in existing}
    fallbacks: list[str] = []
    for token in tokens:
        normalised = _normalise_token(token)
        if not normalised or normalised in STOPWORDS:
            continue
        candidate = normalised.capitalize()
        if candidate.lower() in seen:
            continue
        seen.add(candidate.lower())
        fallbacks.append(candidate)
        if len(fallbacks) >= needed:
            break
    default_index = 0
    while len(fallbacks) < needed:
        placeholder = DEFAULT_FALLBACK_TOPICS[default_index % len(DEFAULT_FALLBACK_TOPICS)]
        if placeholder.lower() not in seen:
            fallbacks.append(placeholder)
            seen.add(placeholder.lower())
        default_index += 1
    return fallbacks


def _actions_for_intent(intent: str, topic_count: int) -> list[str]:
    base_actions = ACTIONS_BY_INTENT.get(intent, ACTIONS_BY_INTENT["how_to"])
    if topic_count <= len(base_actions):
        return base_actions[:topic_count]
    actions = list(base_actions)
    while len(actions) < topic_count:
        actions.append(base_actions[-1])
    return actions


def _enumerate_phrase_candidates(
    tokens: Sequence[str],
    prompt_vector: np.ndarray,
    embeddings: Optional[GloVeEmbeddings],
    knowledge: Optional[KnowledgeNetwork],
) -> list[PhraseCandidate]:
    bigram_pmi = _compute_bigram_pmi(tokens)
    threshold = _percentile(list(bigram_pmi.values()), 0.8)
    max_length = min(4, len(tokens))
    seen: dict[str, PhraseCandidate] = {}

    def add_candidate(
        window: tuple[str, ...],
        lemmas: tuple[str, ...],
        tf_override: Optional[int] = None,
    ) -> None:
        candidate = _build_candidate(
            window,
            lemmas,
            prompt_vector,
            embeddings,
            knowledge,
            tokens,
            bigram_pmi,
            tf_override=tf_override,
        )
        if candidate is None:
            return
        existing = seen.get(candidate.text)
        if existing is None or candidate.score > existing.score:
            seen[candidate.text] = candidate

    for length in range(1, max_length + 1):
        for start in range(0, len(tokens) - length + 1):
            window = tuple(tokens[start : start + length])
            lemmas = tuple(_lemmatise_token(token) for token in window)
            if any(lemma in STOPWORDS for lemma in lemmas):
                continue
            if length > 1 and not _passes_pmi(window, bigram_pmi, threshold):
                continue
            add_candidate(window, lemmas)

    for base_phrase, expansions in PHRASE_EXPANSIONS.items():
        base_tokens = tuple(tokenize(base_phrase))
        if not _contains_sequence(tokens, base_tokens):
            continue
        for extra in expansions:
            extra_tokens = tuple(tokenize(extra))
            lemmas = tuple(_lemmatise_token(token) for token in extra_tokens)
            add_candidate(extra_tokens, lemmas, tf_override=1)
    return list(seen.values())


def _build_candidate(
    window: tuple[str, ...],
    lemmas: tuple[str, ...],
    prompt_vector: np.ndarray,
    embeddings: Optional[GloVeEmbeddings],
    knowledge: Optional[KnowledgeNetwork],
    prompt_tokens: Sequence[str],
    bigram_pmi: dict[tuple[str, str], float],
    tf_override: Optional[int] = None,
) -> Optional[PhraseCandidate]:
    if not window:
        return None
    if lemmas and lemmas[0] in VERB_BLACKLIST:
        return None
    text = " ".join(window)
    embedding = _phrase_embedding(window, embeddings)
    relevance = _cosine_similarity(embedding, prompt_vector)
    tfidf = _tf_idf(window, prompt_tokens, knowledge, tf_override=tf_override)
    bonus = _pmi_bonus(window, bigram_pmi)
    length_reward = LENGTH_BONUS * max(len(window) - 1, 0)
    score = ALPHA * relevance + BETA * tfidf + GAMMA * bonus + length_reward
    return PhraseCandidate(
        tokens=window,
        lemmas=lemmas,
        text=text,
        embedding=embedding,
        relevance=relevance,
        tfidf=tfidf,
        bonus=bonus,
        score=score,
        ngrams=_build_ngram_set(window),
    )


def _select_phrases(
    candidates: Sequence[PhraseCandidate],
    prompt_tokens: Sequence[str],
) -> list[PhraseCandidate]:
    if not candidates:
        return []
    prompt_ngrams = _build_ngram_set(tuple(prompt_tokens))
    remaining = list(candidates)
    selected: list[PhraseCandidate] = []
    while remaining and len(selected) < PHRASE_LIMIT:
        best_score = float("-inf")
        best_candidate: Optional[PhraseCandidate] = None
        for candidate in remaining:
            if any(
                set(candidate.lemmas).issubset(set(chosen.lemmas))
                or set(chosen.lemmas).issubset(set(candidate.lemmas))
                for chosen in selected
            ):
                continue
            mmr = _mmr(candidate, selected)
            overlap = _overlap_penalty(candidate, prompt_ngrams)
            total = mmr + MMR_ETA * candidate.score - OVERLAP_MU * overlap
            if total > best_score:
                best_score = total
                best_candidate = candidate
        if best_candidate is None:
            break
        selected.append(best_candidate)
        remaining.remove(best_candidate)
    has_prompt_collocation = False
    for item in selected:
        if len(item.tokens) > 1 and _contains_sequence(prompt_tokens, item.tokens):
            has_prompt_collocation = True
            break
    if not has_prompt_collocation:
        collocations = [
            candidate
            for candidate in candidates
            if len(candidate.tokens) > 1 and _contains_sequence(prompt_tokens, candidate.tokens)
        ]
        if collocations:
            best_collocation = max(collocations, key=lambda cand: cand.score)
            if best_collocation not in selected:
                selected.append(best_collocation)
                selected = sorted(
                    selected,
                    key=lambda cand: cand.score,
                    reverse=True,
                )[:PHRASE_LIMIT]
    selected.sort(key=lambda cand: cand.score, reverse=True)
    return selected


def _phrase_embedding(
    tokens: Sequence[str],
    embeddings: Optional[GloVeEmbeddings],
) -> NDArray[np.float64]:
    if embeddings is None:
        return cast(NDArray[np.float64], np.zeros((0,), dtype=float))
    vectors = embeddings.encode_tokens(tokens)
    if vectors.size == 0:
        return cast(
            NDArray[np.float64],
            np.zeros((embeddings.config.dimension,), dtype=float),
        )
    return cast(NDArray[np.float64], np.mean(vectors, axis=0))


def _tf_idf(
    phrase: Sequence[str],
    prompt_tokens: Sequence[str],
    knowledge: Optional[KnowledgeNetwork],
    tf_override: Optional[int] = None,
) -> float:
    tf = tf_override if tf_override is not None else _term_frequency(phrase, prompt_tokens)
    if tf == 0:
        return 0.0
    if knowledge is None or not getattr(knowledge, "entities", None):
        corpus_size = 1
        df = 1
    else:
        corpus_size = 1 + len(knowledge.entities)
        phrase_text = " ".join(phrase)
        df = 1
        for entity in knowledge.entities:
            entity_norm = " ".join(tokenize(entity))
            if phrase_text in entity_norm:
                df += 1
    return float(tf * math.log((corpus_size + 1e-6) / (df + 1e-6)))


def _term_frequency(phrase: Sequence[str], tokens: Sequence[str]) -> int:
    length = len(phrase)
    if length == 0 or length > len(tokens):
        return 0
    count = 0
    for start in range(0, len(tokens) - length + 1):
        if tuple(tokens[start : start + length]) == tuple(phrase):
            count += 1
    return count


def _pmi_bonus(
    phrase: Sequence[str],
    bigram_pmi: dict[tuple[str, str], float],
) -> float:
    if len(phrase) <= 1:
        return 0.0
    values = [bigram_pmi.get((phrase[i], phrase[i + 1]), 0.0) for i in range(len(phrase) - 1)]
    if not values:
        return 0.0
    average = sum(values) / len(values)
    return float(min(average, PMI_BONUS_CAP))


def _passes_pmi(
    phrase: Sequence[str],
    bigram_pmi: dict[tuple[str, str], float],
    threshold: float,
) -> bool:
    if len(phrase) <= 1:
        return True
    for i in range(len(phrase) - 1):
        if bigram_pmi.get((phrase[i], phrase[i + 1]), float("-inf")) < threshold:
            return False
    return True


def _compute_bigram_pmi(tokens: Sequence[str]) -> dict[tuple[str, str], float]:
    if len(tokens) < 2:
        return {}
    unigram_counts: Counter[str] = Counter(tokens)
    bigram_counts: Counter[tuple[str, str]] = Counter(
        (tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)
    )
    total_tokens = sum(unigram_counts.values())
    total_bigrams = max(sum(bigram_counts.values()), 1)
    result: dict[tuple[str, str], float] = {}
    for bigram, count in bigram_counts.items():
        p_bigram = (count + 1e-12) / total_bigrams
        p_first = (unigram_counts[bigram[0]] + 1e-12) / total_tokens
        p_second = (unigram_counts[bigram[1]] + 1e-12) / total_tokens
        result[bigram] = math.log(p_bigram / (p_first * p_second))
    return result


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return float("-inf")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = int(math.ceil(percentile * (len(ordered) - 1)))
    return ordered[index]


def _mmr(candidate: PhraseCandidate, selected: Sequence[PhraseCandidate]) -> float:
    if not selected:
        return candidate.relevance
    max_similarity = max(
        _cosine_similarity(candidate.embedding, other.embedding) for other in selected
    )
    return MMR_LAMBDA * candidate.relevance - (1 - MMR_LAMBDA) * max_similarity


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    if vec_a.size == 0 or vec_b.size == 0:
        return 0.0
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def _overlap_penalty(
    candidate: PhraseCandidate,
    prompt_ngrams: set[tuple[str, ...]],
) -> float:
    if not candidate.ngrams:
        return 0.0
    overlap = candidate.ngrams & prompt_ngrams
    return len(overlap) / len(candidate.ngrams)


def _build_ngram_set(tokens: Sequence[str]) -> set[tuple[str, ...]]:
    items = list(tokens)
    ngrams: set[tuple[str, ...]] = set()
    for length in range(1, len(items) + 1):
        for start in range(0, len(items) - length + 1):
            ngrams.add(tuple(items[start : start + length]))
    return ngrams


def _format_phrase(tokens: Sequence[str]) -> str:
    return " ".join(token.capitalize() for token in tokens)


def _lemmatise_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 3 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _contains_sequence(tokens: Sequence[str], pattern: Sequence[str]) -> bool:
    if not pattern:
        return False
    length = len(pattern)
    for start in range(0, len(tokens) - length + 1):
        if tuple(tokens[start : start + length]) == tuple(pattern):
            return True
    return False


def _build_related_topics(
    knowledge: Optional[KnowledgeNetwork], phrases: Sequence[str], tokens: Iterable[str]
) -> tuple[str, list[str]]:
    if knowledge is None or not getattr(knowledge, "entities", None):
        return "", []
    token_list = list(tokens)
    prompt_text = " ".join(token_list).lower()
    candidate: Optional[str] = None
    for entity_name in getattr(knowledge, "entities", {}):
        if entity_name.lower() in prompt_text:
            candidate = entity_name
            break
    if candidate is None and phrases:
        candidate = _match_knowledge_entity(knowledge, phrases[0])
    if candidate is None and token_list:
        candidate = _match_knowledge_entity(
            knowledge,
            _normalise_token(token_list[-1]),
        )
    if candidate is None:
        return "", []
    neighbours = knowledge.neighbours(candidate, top_k=3)
    if not neighbours:
        return "", []
    related_topics = ", ".join(name for name, _ in neighbours)
    hits = [f"{candidate}->{name}" for name, _ in neighbours]
    return f"Related concepts worth exploring: {related_topics}.", hits


def _match_knowledge_entity(
    knowledge: KnowledgeNetwork,
    phrase: str,
) -> Optional[str]:
    phrase_tokens = tuple(tokenize(phrase))
    if not phrase_tokens:
        return None
    for entity in knowledge.entities:
        entity_tokens = tuple(tokenize(entity))
        if entity_tokens == phrase_tokens:
            return entity
    for entity in knowledge.entities:
        entity_tokens = tuple(tokenize(entity))
        if all(token in entity_tokens for token in phrase_tokens):
            return entity
    return None

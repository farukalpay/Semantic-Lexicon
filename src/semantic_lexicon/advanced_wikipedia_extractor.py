# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""Advanced Wikipedia extraction with topic coherence and knowledge graph building."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import requests

from .logging import configure_logging

if TYPE_CHECKING:
    import numpy as np

LOGGER = configure_logging(logger_name=__name__)

# Wikipedia API configuration
WIKIPEDIA_API_URL = "https://en.wikipedia.org/api/rest_v1"
WIKIPEDIA_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "SemanticLexicon/2.0 (https://github.com/semantic-lexicon)"


@dataclass
class ConceptNode:
    """Represents a concept in the knowledge graph."""

    name: str
    source_page: str
    depth: int
    relevance_score: float
    connections: set[str] = field(default_factory=set)
    attributes: dict[str, Any] = field(default_factory=dict)
    wikipedia_categories: list[str] = field(default_factory=list)


@dataclass
class KnowledgeGraph:
    """Dynamic knowledge graph built from Wikipedia."""

    nodes: dict[str, ConceptNode] = field(default_factory=dict)
    edges: list[tuple[str, str, float]] = field(default_factory=list)
    topic_centroid: Optional[np.ndarray] = None

    def add_node(self, node: ConceptNode):
        self.nodes[node.name] = node

    def add_edge(self, from_node: str, to_node: str, weight: float = 1.0):
        self.edges.append((from_node, to_node, weight))
        if from_node in self.nodes:
            self.nodes[from_node].connections.add(to_node)
        if to_node in self.nodes:
            self.nodes[to_node].connections.add(from_node)

    def get_connected_concepts(self, concept: str, max_distance: int = 2) -> set[str]:
        """Get concepts within max_distance edges from the given concept."""
        if concept not in self.nodes:
            return set()

        visited = set()
        queue = deque([(concept, 0)])
        connected = set()

        while queue:
            current, distance = queue.popleft()
            if current in visited or distance > max_distance:
                continue

            visited.add(current)
            connected.add(current)

            if current in self.nodes:
                for neighbor in self.nodes[current].connections:
                    if neighbor not in visited:
                        queue.append((neighbor, distance + 1))

        return connected


class AdvancedWikipediaExtractor:
    """Advanced Wikipedia extractor with topic coherence and graph building."""

    def __init__(self, max_depth: int = 1, max_pages_per_level: int = 2):
        self.max_depth = max_depth  # Reduced from 3 to 1 for speed
        self.max_pages_per_level = max_pages_per_level  # Reduced from 5 to 2 for speed
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.knowledge_graph = KnowledgeGraph()
        self.visited_pages: set[str] = set()
        self.topic_categories: set[str] = set()

    def build_knowledge_graph(self, topic: str) -> KnowledgeGraph:
        """Build a comprehensive knowledge graph for the topic."""
        self.knowledge_graph = KnowledgeGraph()
        self.visited_pages = set()
        self.topic_categories = set()

        # Start with the main topic
        main_page = self._search_wikipedia(topic)
        if not main_page:
            LOGGER.warning(f"No Wikipedia page found for topic: {topic}")
            return self.knowledge_graph

        # Build graph recursively
        self._explore_topic(main_page, depth=0, parent_relevance=1.0)

        # Calculate semantic coherence
        self._calculate_semantic_coherence()

        # Prune irrelevant nodes
        self._prune_graph()

        return self.knowledge_graph

    def _explore_topic(self, page_title: str, depth: int, parent_relevance: float):
        """Recursively explore Wikipedia pages to build the knowledge graph."""
        if depth > self.max_depth or page_title in self.visited_pages:
            return

        self.visited_pages.add(page_title)

        # Get page data
        page_data = self._get_page_full_data(page_title)
        if not page_data:
            return

        # Create node for this concept
        relevance = parent_relevance * (0.8**depth)  # Decay relevance with depth
        node = ConceptNode(
            name=page_title,
            source_page=page_title,
            depth=depth,
            relevance_score=relevance,
            wikipedia_categories=page_data.get("categories", []),
        )

        # Extract attributes from page
        node.attributes["summary"] = page_data.get("summary", "")
        node.attributes["key_terms"] = self._extract_key_terms(page_data.get("content", ""))

        self.knowledge_graph.add_node(node)

        # Store categories for topic coherence
        if depth == 0:
            self.topic_categories.update(page_data.get("categories", []))

        # Get linked pages
        linked_pages = self._get_linked_pages(page_title)

        # Score and filter linked pages
        scored_links = self._score_linked_pages(linked_pages, page_data.get("categories", []))

        # Explore top linked pages
        for _i, (link_title, link_score) in enumerate(scored_links[: self.max_pages_per_level]):
            if link_title not in self.visited_pages:
                # Add edge
                self.knowledge_graph.add_edge(page_title, link_title, link_score)

                # Recursively explore
                self._explore_topic(link_title, depth + 1, relevance * link_score)

    def _get_page_full_data(self, page_title: str) -> Optional[dict]:
        """Get comprehensive data about a Wikipedia page."""
        params: dict[str, str | bool | int] = {
            "action": "query",
            "format": "json",
            "titles": page_title,
            "prop": "extracts|categories|links|info",
            "exintro": True,
            "explaintext": True,
            "cllimit": 20,
            "pllimit": 50,
        }

        try:
            response = self.session.get(WIKIPEDIA_SEARCH_URL, params=params)
            response.raise_for_status()
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            if not pages:
                return None

            page_data = next(iter(pages.values()))

            # Extract categories
            categories = [
                cat.get("title", "").replace("Category:", "")
                for cat in page_data.get("categories", [])
            ]

            # Extract links
            links = [
                link.get("title", "")
                for link in page_data.get("links", [])
                if not link.get("title", "").startswith(("Template:", "Help:", "Wikipedia:"))
            ]

            return {
                "title": page_data.get("title", ""),
                "summary": page_data.get("extract", "")[:500],
                "content": page_data.get("extract", ""),
                "categories": categories,
                "links": links,
            }

        except Exception as e:
            LOGGER.error(f"Failed to get data for '{page_title}': {e}")
            return None

    def _get_linked_pages(self, page_title: str) -> list[str]:
        """Get pages linked from this Wikipedia page."""
        params: dict[str, str | int] = {
            "action": "query",
            "format": "json",
            "titles": page_title,
            "prop": "links",
            "pllimit": 100,
        }

        try:
            response = self.session.get(WIKIPEDIA_SEARCH_URL, params=params)
            response.raise_for_status()
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            if not pages:
                return []

            page_data = next(iter(pages.values()))
            links = page_data.get("links", [])

            # Filter out meta pages
            filtered_links = [
                link.get("title", "")
                for link in links
                if not link.get("title", "").startswith(
                    ("Template:", "Help:", "Wikipedia:", "File:", "Category:")
                )
                and not link.get("title", "").endswith(" (disambiguation)")
            ]

            return filtered_links[:50]  # Limit to top 50 links

        except Exception as e:
            LOGGER.error(f"Failed to get links for '{page_title}': {e}")
            return []

    def _score_linked_pages(
        self, linked_pages: list[str], parent_categories: list[str]
    ) -> list[tuple[str, float]]:
        """Score linked pages based on relevance to the topic."""
        scored = []

        for page in linked_pages:
            score = self._calculate_page_relevance(page, parent_categories)
            scored.append((page, score))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _calculate_page_relevance(self, page_title: str, parent_categories: list[str]) -> float:
        """Calculate relevance score for a potential page to explore."""
        score = 0.5  # Base score

        # Check category overlap
        page_categories = self._get_page_categories(page_title)
        if page_categories:
            category_overlap = len(set(page_categories) & set(parent_categories))
            score += 0.3 * min(category_overlap / max(len(parent_categories), 1), 1.0)

        # Check topic category overlap
        if self.topic_categories:
            topic_overlap = len(set(page_categories) & self.topic_categories)
            score += 0.2 * min(topic_overlap / max(len(self.topic_categories), 1), 1.0)

        return min(score, 1.0)

    def _get_page_categories(self, page_title: str) -> list[str]:
        """Get categories for a Wikipedia page."""
        params: dict[str, str | int] = {
            "action": "query",
            "format": "json",
            "titles": page_title,
            "prop": "categories",
            "cllimit": 10,
        }

        try:
            response = self.session.get(WIKIPEDIA_SEARCH_URL, params=params)
            response.raise_for_status()
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            if not pages:
                return []

            page_data = next(iter(pages.values()))
            categories = [
                cat.get("title", "").replace("Category:", "")
                for cat in page_data.get("categories", [])
            ]

            return categories

        except Exception:
            return []

    def _extract_key_terms(self, content: str) -> list[str]:
        """Extract key technical terms from content."""
        # Extract capitalized phrases (likely proper nouns/technical terms)
        terms = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", content)

        # Extract terms with specific patterns
        technical_patterns = [
            r"\b\w+(?:ation|ization|ment|ance|ence|ity|ism|ology|graphy)\b",
            r"\b\w+(?:neural|network|algorithm|model|system|theory)\b",
        ]

        for pattern in technical_patterns:
            terms.extend(re.findall(pattern, content, re.IGNORECASE))

        # Remove duplicates and return top terms
        term_counts: dict[str, int] = {}
        for term in terms:
            term_lower = term.lower()
            term_counts[term_lower] = term_counts.get(term_lower, 0) + 1

        # Sort by frequency
        sorted_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
        return [term for term, _ in sorted_terms[:20]]

    def _calculate_semantic_coherence(self):
        """Calculate semantic coherence scores for all nodes."""
        # Use PageRank-style algorithm
        if not self.knowledge_graph.nodes:
            return

        # Initialize scores
        scores = {name: node.relevance_score for name, node in self.knowledge_graph.nodes.items()}

        # Iterate to propagate relevance
        for _ in range(10):  # Fixed iterations
            new_scores = {}
            for name, node in self.knowledge_graph.nodes.items():
                # Base score
                score = node.relevance_score * 0.5

                # Add contribution from connected nodes
                for connection in node.connections:
                    if connection in scores:
                        score += scores[connection] * 0.1

                new_scores[name] = min(score, 1.0)

            scores = new_scores

        # Update node scores
        for name, score in scores.items():
            if name in self.knowledge_graph.nodes:
                self.knowledge_graph.nodes[name].relevance_score = score

    def _prune_graph(self, min_relevance: float = 0.1):
        """Remove nodes with low relevance scores."""
        nodes_to_remove = [
            name
            for name, node in self.knowledge_graph.nodes.items()
            if node.relevance_score < min_relevance
        ]

        for name in nodes_to_remove:
            del self.knowledge_graph.nodes[name]

        # Remove edges involving removed nodes
        self.knowledge_graph.edges = [
            (from_node, to_node, weight)
            for from_node, to_node, weight in self.knowledge_graph.edges
            if from_node not in nodes_to_remove and to_node not in nodes_to_remove
        ]

    def _search_wikipedia(self, query: str) -> Optional[str]:
        """Search Wikipedia and return the most relevant page title."""
        params: dict[str, str | int] = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": 1,
        }

        try:
            response = self.session.get(WIKIPEDIA_SEARCH_URL, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("query", {}).get("search"):
                title = data["query"]["search"][0]["title"]
                return str(title) if title else None
            return None

        except Exception as e:
            LOGGER.error(f"Wikipedia search failed for '{query}': {e}")
            return None

    def get_relevant_concepts(self, topic: str, limit: int = 10) -> list[dict[str, Any]]:
        """Get the most relevant concepts for a topic."""
        # Build knowledge graph if not already built
        if not self.knowledge_graph.nodes:
            self.build_knowledge_graph(topic)

        # Sort nodes by relevance
        sorted_nodes = sorted(
            self.knowledge_graph.nodes.values(), key=lambda x: x.relevance_score, reverse=True
        )

        # Convert to output format
        concepts = []
        for node in sorted_nodes[:limit]:
            concepts.append(
                {
                    "name": node.name,
                    "relevance": node.relevance_score,
                    "depth": node.depth,
                    "key_terms": node.attributes.get("key_terms", []),
                    "connections": list(node.connections)[:5],
                }
            )

        return concepts


class TopicCoherenceManager:
    """Manages topic coherence and prevents topic drift."""

    def __init__(self):
        self.topic_history = []
        self.topic_embeddings = {}
        self.drift_threshold = 0.3

    def is_coherent(self, new_concept: str, topic_context: list[str]) -> bool:
        """Check if a new concept is coherent with the topic context."""
        if not topic_context:
            return True

        # Simple heuristic: check word overlap
        new_words = set(new_concept.lower().split())
        context_words = set()
        for context_item in topic_context:
            context_words.update(context_item.lower().split())

        # Remove common words
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
        new_words -= common_words
        context_words -= common_words

        if not new_words or not context_words:
            return True

        # Calculate Jaccard similarity
        intersection = len(new_words & context_words)
        union = len(new_words | context_words)

        similarity = intersection / union if union > 0 else 0

        return bool(similarity > self.drift_threshold)

    def filter_concepts(self, concepts: list[str], topic: str) -> list[str]:
        """Filter concepts to maintain topic coherence."""
        set(topic.lower().split())
        filtered: list[str] = []

        for concept in concepts:
            if self.is_coherent(concept, [topic] + filtered):
                filtered.append(concept)

        return filtered

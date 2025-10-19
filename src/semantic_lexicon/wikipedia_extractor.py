# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""Wikipedia term extraction for knowledge augmentation."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote

import requests

from .logging import configure_logging

LOGGER = configure_logging(logger_name=__name__)

# Wikipedia API configuration
WIKIPEDIA_API_URL = "https://en.wikipedia.org/api/rest_v1"
WIKIPEDIA_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "SemanticLexicon/1.0 (https://github.com/semantic-lexicon)"


@dataclass
class WikipediaTerm:
    """Represents an extracted term from Wikipedia."""

    term: str
    context: str  # The type of information (definition, date, number, etc.)
    source_title: str
    confidence: float


@dataclass
class WikipediaFact:
    """Represents an atomic fact extracted from Wikipedia."""

    subject: str
    predicate: str
    object: str
    context: str
    confidence: float
    source_title: str


class WikipediaTermExtractor:
    """Extracts individual terms and facts from Wikipedia."""

    def __init__(self, cache_size: int = 1000):
        self.cache: dict[str, list[WikipediaTerm]] = {}
        self.fact_cache: dict[str, list[WikipediaFact]] = {}
        self.cache_size = cache_size
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def extract_terms_for_topic(self, topic: str) -> list[WikipediaTerm]:
        """Extract relevant terms for a given topic."""
        # Check cache first
        cache_key = topic.lower()
        if cache_key in self.cache:
            LOGGER.debug(f"Cache hit for topic: {topic}")
            return self.cache[cache_key]

        # Search Wikipedia for the topic
        page_title = self._search_wikipedia(topic)
        if not page_title:
            LOGGER.warning(f"No Wikipedia page found for topic: {topic}")
            return []

        # Get page summary and extract terms
        terms = self._extract_from_page(page_title, topic)

        # Update cache
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            self.cache.pop(next(iter(self.cache)))
        self.cache[cache_key] = terms

        return terms

    def _search_wikipedia(self, query: str) -> Optional[str]:
        """Search Wikipedia and return the most relevant page title."""
        params = {
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
                return data["query"]["search"][0]["title"]
            return None

        except Exception as e:
            LOGGER.error(f"Wikipedia search failed for '{query}': {e}")
            return None

    def _extract_from_page(self, page_title: str, original_topic: str) -> list[WikipediaTerm]:
        """Extract terms from a Wikipedia page."""
        terms = []

        # Get page summary
        summary = self._get_page_summary(page_title)
        if summary:
            # Extract key terms from summary
            extracted = self._extract_terms_from_text(summary, page_title, original_topic)
            terms.extend(extracted)

        # Get page sections for more detailed information
        sections = self._get_page_sections(page_title)
        for section in sections[:3]:  # Limit to first 3 sections
            section_terms = self._extract_terms_from_text(
                section.get("text", ""), page_title, original_topic
            )
            terms.extend(section_terms)

        return terms

    def _get_page_summary(self, page_title: str) -> Optional[str]:
        """Get the summary of a Wikipedia page."""
        url = f"{WIKIPEDIA_API_URL}/page/summary/{quote(page_title)}"

        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            return data.get("extract", "")

        except Exception as e:
            LOGGER.error(f"Failed to get summary for '{page_title}': {e}")
            return None

    def _get_page_sections(self, page_title: str) -> list[dict]:
        """Get sections from a Wikipedia page."""
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "titles": page_title,
            "exintro": False,
            "explaintext": True,
            "exsectionformat": "plain",
        }

        try:
            response = self.session.get(WIKIPEDIA_SEARCH_URL, params=params)
            response.raise_for_status()
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            if pages:
                page_data = next(iter(pages.values()))
                extract = page_data.get("extract", "")

                # Simple section parsing
                sections = []
                current_section = {"title": "Introduction", "text": ""}

                for line in extract.split("\n"):
                    if line.startswith("==") and line.endswith("=="):
                        if current_section["text"]:
                            sections.append(current_section)
                        current_section = {"title": line.strip("= "), "text": ""}
                    else:
                        current_section["text"] += line + " "

                if current_section["text"]:
                    sections.append(current_section)

                return sections[:5]  # Return first 5 sections

            return []

        except Exception as e:
            LOGGER.error(f"Failed to get sections for '{page_title}': {e}")
            return []

    def extract_facts_for_topic(self, topic: str) -> list[WikipediaFact]:
        """Extract atomic facts for a given topic."""
        cache_key = f"facts_{topic.lower()}"
        if cache_key in self.fact_cache:
            return self.fact_cache[cache_key]

        # Search Wikipedia for the topic
        page_title = self._search_wikipedia(topic)
        if not page_title:
            LOGGER.warning(f"No Wikipedia page found for topic: {topic}")
            return []

        # Get page content and extract facts
        facts = self._extract_facts_from_page(page_title, topic)

        # Cache the facts
        if len(self.fact_cache) >= self.cache_size:
            self.fact_cache.pop(next(iter(self.fact_cache)))
        self.fact_cache[cache_key] = facts

        return facts

    def _extract_facts_from_page(self, page_title: str, topic: str) -> list[WikipediaFact]:
        """Extract atomic facts from a Wikipedia page."""
        facts = []

        # Get page summary and sections
        summary = self._get_page_summary(page_title)
        if summary:
            facts.extend(self._extract_facts_from_text(summary, page_title, topic))

        sections = self._get_page_sections(page_title)
        for section in sections[:3]:
            section_facts = self._extract_facts_from_text(
                section.get("text", ""), page_title, topic
            )
            facts.extend(section_facts)

        return facts

    def _extract_facts_from_text(
        self, text: str, source_title: str, topic: str
    ) -> list[WikipediaFact]:
        """Extract atomic facts from text."""
        facts = []

        if not text:
            return facts

        sentences = text.split(".")
        topic_normalized = topic.lower().replace("_", " ")

        for sentence in sentences[:10]:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Extract definition facts - capture complete definitions
            if " is a " in sentence.lower() or " is an " in sentence.lower():
                # Try to capture the full definition
                # First, try to get everything up to a period
                for pattern in [
                    r"([^,]+?)\s+is\s+an?\s+([^\.]+)",  # Everything until period
                    # Until conjunction
                    r"([^,]+?)\s+is\s+an?\s+([^,]+?)(?:,\s*(?:which|that|and))",
                ]:
                    match = re.search(pattern, sentence, re.IGNORECASE)
                    if match:
                        subject = match.group(1).strip()
                        obj = match.group(2).strip()

                        # Only use if the subject is relevant to our topic
                        if (
                            topic_normalized in subject.lower()
                            or subject.lower() in ["it", "this"]
                            or len(subject.split()) <= 3
                        ):
                            # Clean and validate the object
                            clean_obj = self._clean_object(obj)

                            # Make sure we got a meaningful definition
                            # At least 3 words for a meaningful definition
                            if len(clean_obj.split()) >= 3:
                                facts.append(
                                    WikipediaFact(
                                        subject=self._clean_subject(subject, topic),
                                        predicate="is_type_of",
                                        object=clean_obj,
                                        context="definition",
                                        confidence=0.95,
                                        source_title=source_title,
                                    )
                                )
                                break

            # Extract "refers to" facts
            if "refers to" in sentence:
                match = re.search(r"([^,]+?)\s+refers to\s+([^,\.]+)", sentence, re.IGNORECASE)
                if match:
                    facts.append(
                        WikipediaFact(
                            subject=self._clean_subject(match.group(1).strip(), topic),
                            predicate="refers_to",
                            object=self._clean_object(match.group(2).strip()),
                            context="definition",
                            confidence=0.9,
                            source_title=source_title,
                        )
                    )

            # Extract capability facts - capture complete capabilities
            if any(
                word in sentence.lower()
                for word in ["can", "enables", "allows", "used to", "used for"]
            ):
                patterns = [
                    r"([^,]+?)\s+can\s+([^\.]+)",  # Full capability until period
                    r"([^,]+?)\s+enables\s+([^\.]+)",
                    r"([^,]+?)\s+allows\s+([^\.]+)",
                    r"([^,]+?)\s+(?:is|are)\s+used\s+(?:to|for)\s+([^\.]+)",
                ]

                for pattern in patterns:
                    match = re.search(pattern, sentence, re.IGNORECASE)
                    if match:
                        subject = self._clean_subject(match.group(1).strip(), topic)
                        capability = self._clean_object(match.group(2).strip())

                        # Ensure we have a meaningful capability
                        if len(capability.split()) >= 2:  # At least 2 words
                            # Avoid duplicate capabilities
                            if not any(
                                f.object == capability for f in facts if f.predicate == "enables"
                            ):
                                facts.append(
                                    WikipediaFact(
                                        subject=subject,
                                        predicate="enables",
                                        object=capability,
                                        context="capability",
                                        confidence=0.85,
                                        source_title=source_title,
                                    )
                                )
                            break

            # Extract composition facts
            if "consists of" in sentence or "composed of" in sentence or "includes" in sentence:
                for pattern in [
                    r"([^,]+?)\s+consists of\s+([^\.]+)",
                    r"([^,]+?)\s+composed of\s+([^\.]+)",
                    r"([^,]+?)\s+includes\s+([^\.]+)",
                ]:
                    match = re.search(pattern, sentence, re.IGNORECASE)
                    if match:
                        facts.append(
                            WikipediaFact(
                                subject=self._clean_subject(match.group(1).strip(), topic),
                                predicate="composed_of",
                                object=self._clean_object(match.group(2).strip()),
                                context="structure",
                                confidence=0.85,
                                source_title=source_title,
                            )
                        )
                        break

            # Extract temporal facts
            year_match = re.search(r"in\s+(1[0-9]{3}|20[0-2][0-9])", sentence)
            if year_match:
                year = year_match.group(1)
                # Look for what happened in that year
                if "introduced" in sentence or "developed" in sentence or "created" in sentence:
                    facts.append(
                        WikipediaFact(
                            subject=topic,
                            predicate="developed_in",
                            object=year,
                            context="temporal",
                            confidence=0.9,
                            source_title=source_title,
                        )
                    )

            # Extract inventor/creator facts
            by_match = re.search(
                r"(?:developed|created|introduced|designed)\s+by\s+([A-Z][^,\.]+)", sentence
            )
            if by_match:
                creator = by_match.group(1).strip()
                facts.append(
                    WikipediaFact(
                        subject=topic,
                        predicate="created_by",
                        object=creator,
                        context="attribution",
                        confidence=0.85,
                        source_title=source_title,
                    )
                )

        return facts

    def _clean_subject(self, subject: str, topic: str) -> str:
        """Clean and normalize subject string."""
        # Remove articles
        subject = re.sub(r"^(the|a|an)\s+", "", subject, flags=re.IGNORECASE)
        # If subject is a pronoun or too generic, use the topic
        if subject.lower() in ["it", "this", "that", "they", "these", "those"]:
            return topic
        return subject.strip()

    def _clean_object(self, obj: str) -> str:
        """Clean and normalize object string."""
        # Remove articles and clean up
        obj = re.sub(r"^(the|a|an)\s+", "", obj, flags=re.IGNORECASE)
        obj = re.sub(r"\s+", " ", obj)

        # Try to preserve complete phrases by looking for sentence endings
        # Don't cut in the middle of a thought
        words = obj.split()
        if len(words) > 15:
            # Look for a natural break point
            for i in range(10, min(15, len(words))):
                word = words[i]
                # If we find a conjunction or preposition, break before it
                if word.lower() in ["and", "or", "but", "which", "that", "with", "for"]:
                    obj = " ".join(words[:i])
                    break
            else:
                # If no natural break, take first 15 words
                obj = " ".join(words[:15])

        # Ensure it doesn't end mid-phrase
        obj = re.sub(r"\s+(of|the|a|an|in|on|at|to|for)$", "", obj)

        return obj.strip()

    def _extract_terms_from_text(
        self, text: str, source_title: str, topic: str
    ) -> list[WikipediaTerm]:
        """Extract individual terms from text - kept for backward compatibility."""
        terms = []

        if not text:
            return terms

        # Extract key concepts as terms (simplified version)
        # This is now secondary to fact extraction
        facts = self._extract_facts_from_text(text, source_title, topic)

        # Convert some facts to terms for backward compatibility
        for fact in facts[:5]:
            if fact.context == "definition":
                terms.append(
                    WikipediaTerm(
                        term=fact.object,
                        context="definition",
                        source_title=source_title,
                        confidence=fact.confidence,
                    )
                )
            elif fact.predicate == "developed_in":
                terms.append(
                    WikipediaTerm(
                        term=fact.object,
                        context="year",
                        source_title=source_title,
                        confidence=fact.confidence,
                    )
                )
            elif fact.predicate == "created_by":
                terms.append(
                    WikipediaTerm(
                        term=fact.object,
                        context="entity",
                        source_title=source_title,
                        confidence=fact.confidence,
                    )
                )

        return terms


class KnowledgeAugmentedGenerator:
    """Generator that augments responses with Wikipedia terms."""

    def __init__(self, wikipedia_extractor: Optional[WikipediaTermExtractor] = None):
        self.wikipedia = wikipedia_extractor or WikipediaTermExtractor()

    def generate_with_facts(self, prompt: str, base_response: str) -> str:
        """Augment a base response with factual terms from Wikipedia."""
        # Identify topics in the prompt
        topics = self._identify_topics(prompt)

        # Collect terms for all topics
        all_terms = defaultdict(list)
        for topic in topics:
            terms = self.wikipedia.extract_terms_for_topic(topic)
            for term in terms:
                all_terms[term.context].append(term)

        # Augment the response
        augmented = self._augment_response(base_response, all_terms)
        return augmented

    def _identify_topics(self, prompt: str) -> list[str]:
        """Identify key topics from the prompt."""
        # Simple topic extraction - can be made more sophisticated
        topics = []

        # Look for explicit topic indicators
        if "neural network" in prompt.lower():
            topics.append("neural network")
        if "machine learning" in prompt.lower():
            topics.append("machine learning")
        if "deep learning" in prompt.lower():
            topics.append("deep learning")

        # Extract noun phrases as potential topics
        # This is simplified - in production, use NLP libraries
        words = prompt.lower().split()
        for i, word in enumerate(words):
            if word in ["explain", "what", "define", "describe"] and i + 1 < len(words):
                # The next word(s) might be the topic
                if i + 2 < len(words):
                    potential_topic = f"{words[i + 1]} {words[i + 2]}"
                    if not any(
                        w in ["is", "are", "the", "a", "an"] for w in [words[i + 1], words[i + 2]]
                    ):
                        topics.append(potential_topic)
                else:
                    topics.append(words[i + 1])

        return topics[:3]  # Limit to 3 topics

    def _augment_response(self, base_response: str, terms: dict[str, list[WikipediaTerm]]) -> str:
        """Augment response with Wikipedia terms."""
        augmented = base_response

        # If we have a definition, incorporate it
        if terms.get("definition"):
            best_def = max(terms["definition"], key=lambda t: t.confidence)
            # Replace generic phrases with specific definitions
            if "This ties closely to" in augmented:
                augmented = augmented.replace(
                    "This ties closely to",
                    f"This relates to {best_def.term}, which ties closely to",
                )

        # Add years for historical context
        if terms.get("year"):
            years = sorted(terms["year"], key=lambda t: t.term)
            if years:
                year_info = f"(developed in {years[0].term})"
                # Find a good place to insert the year
                if "." in augmented:
                    parts = augmented.split(".", 1)
                    augmented = f"{parts[0]} {year_info}.{parts[1]}"

        # Add technical terms for depth
        if terms.get("technical_term"):
            tech_terms = [t.term for t in terms["technical_term"][:2]]
            if tech_terms:
                terms_str = " and ".join(tech_terms)
                augmented += f" Key concepts include {terms_str}."

        # Add entities for specificity
        if terms.get("entity"):
            entities = [t.term for t in terms["entity"][:2] if t.confidence > 0.75]
            if entities:
                augmented += f" Notable contributors include {', '.join(entities)}."

        return augmented

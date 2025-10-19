# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""Direct Q&A response generation using Wikipedia facts."""

from __future__ import annotations

import random
import re
from typing import Optional

from .logging import configure_logging
from .wikipedia_extractor import WikipediaFact, WikipediaTerm, WikipediaTermExtractor

LOGGER = configure_logging(logger_name=__name__)

# Response templates for natural variation
DEFINITION_TEMPLATES = [
    "{topic} can be understood as {definition}.",
    "{topic} represents {definition}.",
    "In essence, {topic} is {definition}.",
    "Simply put, {topic} refers to {definition}.",
    "{topic} describes {definition}.",
]

CAPABILITY_TEMPLATES = [
    "This {enables} {capability}.",
    "It {enables} {capability}.",
    "One key function is to {enable} {capability}.",
    "{topic} makes it possible to {capability}.",
    "Through {topic}, one can {capability}.",
]

TEMPORAL_TEMPLATES = [
    "The concept emerged in {year}.",
    "Development began around {year}.",
    "This technology dates back to {year}.",
    "First introduced in {year}.",
    "Origins trace back to {year}.",
]

CREATOR_TEMPLATES = [
    "Pioneered by {creator}.",
    "{creator} played a key role in its development.",
    "The work of {creator} was instrumental.",
    "Credit goes to {creator} for this innovation.",
    "{creator} made significant contributions.",
]

COMPOSITION_TEMPLATES = [
    "It consists of {components}.",
    "Key elements include {components}.",
    "The main components are {components}.",
    "It's composed of {components}.",
    "Essential parts include {components}.",
]


class DirectQAGenerator:
    """Generates direct, informative answers using Wikipedia facts."""

    def __init__(self):
        self.wikipedia = WikipediaTermExtractor()

    def generate_answer(self, prompt: str) -> str:
        """Generate a direct answer to the user's question."""
        # Identify the main topic and question type
        topic, question_type = self._analyze_prompt(prompt)

        if not topic:
            return "I need more specific information to provide an accurate answer."

        # Extract Wikipedia facts for the topic
        facts = self.wikipedia.extract_facts_for_topic(topic)

        if not facts:
            # Fallback to term extraction
            terms = self.wikipedia.extract_terms_for_topic(topic)
            if not terms:
                return f"I need more context about {topic} to provide a comprehensive answer."
            return self._build_answer_from_terms(topic, terms, question_type)

        # Build answer based on facts and question type
        return self._synthesize_answer(topic, facts, question_type)

    def _analyze_prompt(self, prompt: str) -> tuple[Optional[str], str]:
        """Analyze prompt to extract topic and question type."""
        prompt_lower = prompt.lower()

        # Determine question type
        question_type = "general"
        if any(word in prompt_lower for word in ["explain", "what is", "what are"]):
            question_type = "explain"
        elif any(phrase in prompt_lower for phrase in ["how does", "how do", "how to", "how can"]):
            question_type = "how"
        elif "why" in prompt_lower:
            question_type = "why"
        elif any(word in prompt_lower for word in ["when", "what year", "what time"]):
            question_type = "when"
        elif any(word in prompt_lower for word in ["who", "whom", "whose"]):
            question_type = "who"

        # Extract topic with improved patterns
        topic = None

        # Enhanced pattern matching for common question formats
        patterns = [
            r"explain\s+(?:the\s+)?(.+?)(?:\?|$)",
            r"what (?:is|are)\s+(?:the\s+)?(.+?)(?:\?|$)",
            r"how (?:do|does|can|could|would|should)\s+(.+?)(?:\?|$)",
            r"why (?:is|are|do|does|should|would)\s+(.+?)(?:\?|$)",
            r"when (?:was|is|are|did|does|will)\s+(.+?)(?:\?|$)",
            r"who (?:is|was|are|were|created|invented|discovered)\s+(.+?)(?:\?|$)",
            r"tell me about\s+(?:the\s+)?(.+?)(?:\?|$)",
            r"describe\s+(?:the\s+)?(.+?)(?:\?|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, prompt_lower, re.IGNORECASE)
            if match:
                topic = match.group(1).strip()
                # Clean up common artifacts
                topic = re.sub(r"\b(the|a|an)\b", "", topic).strip()
                if topic:
                    break

        # Improved fallback: extract noun phrases more intelligently
        if not topic:
            # Remove question words and auxiliary verbs
            words_to_remove = [
                "explain",
                "what",
                "how",
                "why",
                "when",
                "who",
                "which",
                "where",
                "describe",
                "tell",
                "me",
                "about",
                "is",
                "are",
                "was",
                "were",
                "do",
                "does",
                "did",
                "can",
                "could",
                "would",
                "should",
                "will",
            ]
            cleaned = prompt_lower
            for word in words_to_remove:
                cleaned = re.sub(r"\b" + word + r"\b", "", cleaned)

            # Clean up and extract the core topic
            topic = cleaned.strip().strip("?.,!").strip()

            # If still empty, try to extract any noun-like phrase
            if not topic:
                noun_match = re.search(r"\b([a-z]+(?:\s+[a-z]+){0,2})\b", prompt_lower)
                if noun_match:
                    candidate = noun_match.group(1)
                    if candidate not in words_to_remove:
                        topic = candidate

        return topic, question_type

    def _synthesize_answer(self, topic: str, facts: list[WikipediaFact], question_type: str) -> str:
        """Synthesize a natural answer from extracted facts."""
        # Group facts by type
        fact_groups = self._group_facts(facts)

        # Build response based on question type
        if question_type == "explain" or question_type == "what":
            return self._build_explanation_from_facts(topic, fact_groups)
        elif question_type == "how":
            return self._build_how_answer_from_facts(topic, fact_groups)
        elif question_type == "why":
            return self._build_why_answer_from_facts(topic, fact_groups)
        elif question_type == "when":
            return self._build_temporal_answer(topic, fact_groups)
        elif question_type == "who":
            return self._build_attribution_answer(topic, fact_groups)
        else:
            return self._build_general_answer_from_facts(topic, fact_groups)

    def _group_facts(self, facts: list[WikipediaFact]) -> dict[str, list[WikipediaFact]]:
        """Group facts by their predicate type."""
        groups = {
            "definitions": [],
            "capabilities": [],
            "composition": [],
            "temporal": [],
            "attribution": [],
        }

        for fact in facts:
            if fact.predicate in ["is_type_of", "refers_to"]:
                groups["definitions"].append(fact)
            elif fact.predicate == "enables":
                groups["capabilities"].append(fact)
            elif fact.predicate == "composed_of":
                groups["composition"].append(fact)
            elif fact.predicate == "developed_in":
                groups["temporal"].append(fact)
            elif fact.predicate == "created_by":
                groups["attribution"].append(fact)

        return groups

    def _build_explanation_from_facts(self, topic: str, fact_groups: dict) -> str:
        """Build a natural explanation from facts."""
        paragraphs = []

        # Start with definition
        if fact_groups["definitions"]:
            fact = fact_groups["definitions"][0]
            template = random.choice(DEFINITION_TEMPLATES)
            definition_sentence = template.format(
                topic=topic.capitalize(), definition=self._naturalize_phrase(fact.object)
            )
            paragraphs.append(definition_sentence)
        else:
            paragraphs.append(f"{topic.capitalize()} is an important concept worth understanding.")

        # Add capabilities (avoid duplicates)
        if fact_groups["capabilities"]:
            # Get unique capabilities
            seen_capabilities = set()
            unique_capabilities = []
            for f in fact_groups["capabilities"]:
                normalized = self._naturalize_phrase(f.object)
                if normalized not in seen_capabilities:
                    seen_capabilities.add(normalized)
                    unique_capabilities.append(normalized)
                if len(unique_capabilities) >= 2:
                    break

            if len(unique_capabilities) == 1:
                template = random.choice(CAPABILITY_TEMPLATES)
                cap_sentence = template.format(
                    topic=topic,
                    enables="enables",
                    enable="enable",
                    capability=unique_capabilities[0],
                )
                paragraphs.append(cap_sentence)
            elif len(unique_capabilities) >= 2:
                cap_sentence = (
                    f"This enables both {unique_capabilities[0]} and {unique_capabilities[1]}."
                )
                paragraphs.append(cap_sentence)

        # Add historical context
        if fact_groups["temporal"]:
            fact = fact_groups["temporal"][0]
            template = random.choice(TEMPORAL_TEMPLATES)
            temporal_sentence = template.format(year=fact.object)
            paragraphs.append(temporal_sentence)

        # Add attribution
        if fact_groups["attribution"]:
            fact = fact_groups["attribution"][0]
            template = random.choice(CREATOR_TEMPLATES)
            creator_sentence = template.format(creator=fact.object)
            paragraphs.append(creator_sentence)

        # Add composition details
        if fact_groups["composition"]:
            components = [self._naturalize_phrase(f.object) for f in fact_groups["composition"][:3]]
            if components:
                template = random.choice(COMPOSITION_TEMPLATES)
                comp_sentence = template.format(components=self._format_list(components))
                paragraphs.append(comp_sentence)

        return " ".join(paragraphs)

    def _naturalize_phrase(self, phrase: str) -> str:
        """Convert a phrase to more natural language."""
        # Remove excess whitespace
        phrase = re.sub(r"\s+", " ", phrase).strip()

        # Handle common patterns
        if phrase.endswith("ing") and len(phrase) > 10:
            # Likely a gerund phrase - keep as is
            return phrase

        # Make sure it doesn't start with 'a' or 'an' redundantly
        phrase = re.sub(r"^(a|an)\s+(a|an)\s+", r"\1 ", phrase, flags=re.IGNORECASE)

        return phrase

    def _format_list(self, items: list[str]) -> str:
        """Format a list of items naturally."""
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"

    def _build_how_answer(self, topic: str, terms: list[WikipediaTerm]) -> str:
        """Build a how-to answer."""
        sentences = []

        # Start with definition
        definitions = [t for t in terms if t.context == "definition"]
        if definitions:
            sentences.append(
                f"To understand how {topic} works, note that it is {definitions[0].term}."
            )

        # Add process-related terms
        technical = [t for t in terms if t.context == "technical_term"]
        if technical:
            processes = [t.term for t in technical[:2]]
            sentences.append(f"The process involves {' and '.join(processes)}.")

        # Add any quantitative aspects
        quantities = [t for t in terms if t.context == "quantity"]
        if quantities:
            sentences.append(f"This typically operates at scales of {quantities[0].term}.")

        return (
            " ".join(sentences)
            if sentences
            else f"The mechanism of {topic} requires detailed technical explanation."
        )

    def _build_why_answer(self, topic: str, terms: list[WikipediaTerm]) -> str:
        """Build a why answer."""
        sentences = []

        # Start with purpose from definition
        definitions = [t for t in terms if t.context == "definition"]
        if definitions:
            sentences.append(f"{topic.capitalize()} exists because it is {definitions[0].term}.")

        # Add historical motivation
        years = [t for t in terms if t.context == "year"]
        if years:
            sentences.append(
                f"It was developed in {years[0].term} to address specific computational needs."
            )

        # Add benefits from technical terms
        technical = [t for t in terms if t.context == "technical_term"]
        if technical:
            benefits = [t.term for t in technical[:2]]
            sentences.append(f"It enables {' and '.join(benefits)}.")

        return (
            " ".join(sentences)
            if sentences
            else f"The rationale for {topic} stems from practical needs."
        )

    def _build_how_answer_from_facts(self, topic: str, fact_groups: dict) -> str:
        """Build a how-to answer from facts."""
        paragraphs = []

        # Start with understanding what it is
        if fact_groups["definitions"]:
            fact = fact_groups["definitions"][0]
            intro = (
                f"To understand how {topic} works, it helps to know that it "
                f"{self._naturalize_phrase(fact.object)}."
            )
            paragraphs.append(intro)

        # Explain the mechanism through capabilities
        if fact_groups["capabilities"]:
            capabilities = [
                self._naturalize_phrase(f.object) for f in fact_groups["capabilities"][:3]
            ]
            if len(capabilities) == 1:
                paragraphs.append(f"The mechanism works by enabling {capabilities[0]}.")
            elif len(capabilities) == 2:
                paragraphs.append(f"It functions through {capabilities[0]} and {capabilities[1]}.")
            else:
                paragraphs.append(f"The process involves {self._format_list(capabilities)}.")

        # Add composition for structure
        if fact_groups["composition"]:
            components = [self._naturalize_phrase(f.object) for f in fact_groups["composition"][:2]]
            if components:
                paragraphs.append(f"The key components are {self._format_list(components)}.")

        # Add any technical context
        if not fact_groups["capabilities"] and not fact_groups["composition"]:
            paragraphs.append(
                f"The specific workings of {topic} depend on the implementation context."
            )

        return " ".join(paragraphs)

    def _build_why_answer_from_facts(self, topic: str, fact_groups: dict) -> str:
        """Build a why answer from facts."""
        paragraphs = []

        # Start with the purpose based on definition
        if fact_groups["definitions"]:
            fact = fact_groups["definitions"][0]
            paragraphs.append(
                f"{topic.capitalize()} exists because it provides "
                f"{self._naturalize_phrase(fact.object)}."
            )

        # Explain benefits through capabilities
        if fact_groups["capabilities"]:
            capabilities = [
                self._naturalize_phrase(f.object) for f in fact_groups["capabilities"][:2]
            ]
            if capabilities:
                reason = f"This is important because it enables {self._format_list(capabilities)}."
                paragraphs.append(reason)

        # Add historical motivation
        if fact_groups["temporal"]:
            year = fact_groups["temporal"][0].object
            if fact_groups["attribution"]:
                creator = fact_groups["attribution"][0].object
                paragraphs.append(
                    f"It was developed in {year} by {creator} to address emerging needs."
                )
            else:
                paragraphs.append(f"It emerged in {year} to solve specific challenges of that era.")

        # If we don't have much info, provide generic reasoning
        if len(paragraphs) == 0:
            paragraphs.append(
                f"The rationale for {topic} comes from practical requirements in the field."
            )

        return " ".join(paragraphs)

    def _build_temporal_answer(self, topic: str, fact_groups: dict) -> str:
        """Build a when/temporal answer from facts."""
        paragraphs = []

        if fact_groups["temporal"]:
            year = fact_groups["temporal"][0].object
            main_sentence = f"{topic.capitalize()} was developed in {year}."
            paragraphs.append(main_sentence)

            # Add creator if available
            if fact_groups["attribution"]:
                creator = fact_groups["attribution"][0].object
                paragraphs.append(f"This was the work of {creator}.")

            # Add context about what it enabled
            if fact_groups["capabilities"]:
                capability = self._naturalize_phrase(fact_groups["capabilities"][0].object)
                paragraphs.append(f"This timing was significant as it enabled {capability}.")
        else:
            # No temporal facts, try to infer from other information
            if fact_groups["attribution"]:
                creator = fact_groups["attribution"][0].object
                paragraphs.append(
                    f"While the exact date is unclear, {topic} is associated with {creator}."
                )
            else:
                paragraphs.append(
                    f"The specific timeline for {topic} would require more historical context."
                )

        return " ".join(paragraphs)

    def _build_attribution_answer(self, topic: str, fact_groups: dict) -> str:
        """Build a who/attribution answer from facts."""
        paragraphs = []

        if fact_groups["attribution"]:
            creator = fact_groups["attribution"][0].object
            main_sentence = f"{creator} is credited with {topic}."
            paragraphs.append(main_sentence)

            # Add temporal context if available
            if fact_groups["temporal"]:
                year = fact_groups["temporal"][0].object
                paragraphs.append(f"This work dates back to {year}.")

            # Add what was created
            if fact_groups["definitions"]:
                definition = self._naturalize_phrase(fact_groups["definitions"][0].object)
                paragraphs.append(f"They developed this as {definition}.")
        else:
            # No attribution facts
            if fact_groups["temporal"]:
                year = fact_groups["temporal"][0].object
                paragraphs.append(
                    f"While {topic} emerged around {year}, specific attribution is unclear."
                )
            else:
                paragraphs.append(
                    f"The specific creators of {topic} would require more historical research."
                )

        return " ".join(paragraphs)

    def _build_general_answer_from_facts(self, topic: str, fact_groups: dict) -> str:
        """Build a general answer from facts - similar to explanation but more concise."""
        return self._build_explanation_from_facts(topic, fact_groups)

    def _build_answer_from_terms(
        self, topic: str, terms: list[WikipediaTerm], question_type: str
    ) -> str:
        """Fallback to build answer from terms when facts aren't available."""
        sentences = []

        # Group terms by context
        term_groups = {
            "definitions": [t for t in terms if t.context == "definition"],
            "years": [t for t in terms if t.context == "year"],
            "entities": [t for t in terms if t.context == "entity"],
            "technical": [t for t in terms if t.context == "technical_term"],
            "quantities": [t for t in terms if t.context == "quantity"],
        }

        # Build response based on question type
        if question_type in ["explain", "what"]:
            # Start with definition
            if term_groups["definitions"]:
                definition = term_groups["definitions"][0].term
                sentences.append(f"{topic.capitalize()} represents {definition}.")
            else:
                sentences.append(f"{topic.capitalize()} is a significant concept in the field.")

            # Add technical details
            if term_groups["technical"]:
                tech_terms = [t.term for t in term_groups["technical"][:2]]
                sentences.append(f"Key aspects involve {self._format_list(tech_terms)}.")

            # Add temporal context
            if term_groups["years"]:
                year = term_groups["years"][0].term
                sentences.append(f"Development began around {year}.")

        elif question_type == "how":
            sentences.append(f"Understanding how {topic} works requires examining its components.")

            if term_groups["technical"]:
                processes = [t.term for t in term_groups["technical"][:2]]
                sentences.append(f"The mechanism involves {self._format_list(processes)}.")

            if term_groups["quantities"]:
                scale = term_groups["quantities"][0].term
                sentences.append(f"Operations typically occur at {scale} scale.")

        elif question_type == "why":
            if term_groups["definitions"]:
                purpose = term_groups["definitions"][0].term
                sentences.append(f"{topic.capitalize()} serves the purpose of {purpose}.")

            if term_groups["years"]:
                year = term_groups["years"][0].term
                sentences.append(f"It emerged in {year} to address specific needs.")

        elif question_type == "when":
            if term_groups["years"]:
                year = term_groups["years"][0].term
                sentences.append(f"{topic.capitalize()} dates back to {year}.")
            else:
                sentences.append(f"The timeline of {topic} requires more historical context.")

        elif question_type == "who":
            if term_groups["entities"]:
                names = [t.term for t in term_groups["entities"][:2]]
                sentences.append(f"Key figures include {self._format_list(names)}.")
            else:
                sentences.append(f"Attribution for {topic} involves multiple contributors.")

        else:
            # General answer
            if term_groups["definitions"]:
                sentences.append(
                    f"{topic.capitalize()} can be described as "
                    f"{term_groups['definitions'][0].term}."
                )
            if term_groups["technical"]:
                concepts = [t.term for t in term_groups["technical"][:2]]
                sentences.append(f"Important aspects include {self._format_list(concepts)}.")

        return " ".join(sentences) if sentences else f"Information about {topic} is being compiled."

    def _build_general_answer(self, topic: str, terms: list[WikipediaTerm]) -> str:
        """Build a general informative answer (legacy method for compatibility)."""
        return self._build_answer_from_terms(topic, terms, "general")

    def _build_explanation(self, topic: str, terms: list[WikipediaTerm]) -> str:
        """Build an explanatory answer (legacy method for compatibility)."""
        return self._build_answer_from_terms(topic, terms, "explain")


class EnhancedSentenceBuilder:
    """Builds sentences word-by-word using Wikipedia facts."""

    def __init__(self):
        self.wikipedia = WikipediaTermExtractor()
        self.sentence_templates = {
            "definition": [
                "{topic} is {definition}, which {property}",
                "A {topic} refers to {definition} that {function}",
                "{topic} can be defined as {definition} used for {purpose}",
            ],
            "historical": [
                "{topic} was first developed in {year} by {person}",
                "The concept of {topic} emerged in {year}",
                "In {year}, {person} introduced {topic}",
            ],
            "technical": [
                "{topic} uses {technique1} and {technique2}",
                "The main components are {component1} and {component2}",
                "{topic} involves {process1} followed by {process2}",
            ],
        }

    def build_response(self, prompt: str) -> str:
        """Build response sentence by sentence, filling in facts."""
        topic = self._extract_topic(prompt)
        if not topic:
            return "Please specify what you'd like to know about."

        # Get Wikipedia facts
        terms = self.wikipedia.extract_terms_for_topic(topic)
        if not terms:
            return f"I don't have enough information about {topic} to provide a detailed answer."

        # Organize terms by type
        facts = self._organize_facts(terms)

        # Build sentences
        sentences = []

        # Definition sentence
        if facts["definitions"]:
            sentence = self._build_definition_sentence(topic, facts)
            sentences.append(sentence)

        # Historical sentence
        if facts["years"] or facts["entities"]:
            sentence = self._build_historical_sentence(topic, facts)
            if sentence:
                sentences.append(sentence)

        # Technical details sentence
        if facts["technical"]:
            sentence = self._build_technical_sentence(topic, facts)
            sentences.append(sentence)

        # Application sentence
        if facts["quantities"] or facts["technical"]:
            sentence = self._build_application_sentence(topic, facts)
            if sentence:
                sentences.append(sentence)

        return " ".join(sentences) if sentences else f"Information about {topic} is limited."

    def _extract_topic(self, prompt: str) -> Optional[str]:
        """Extract the main topic from the prompt."""
        prompt_lower = prompt.lower()

        # Remove common question words
        for word in ["explain", "what", "how", "why", "describe", "tell me about", "is", "are"]:
            prompt_lower = prompt_lower.replace(word, "")

        # Clean up
        topic = prompt_lower.strip().strip("?").strip()
        return topic if topic else None

    def _organize_facts(self, terms: list[WikipediaTerm]) -> dict[str, list[WikipediaTerm]]:
        """Organize terms by their context type."""
        facts = {
            "definitions": [],
            "years": [],
            "entities": [],
            "technical": [],
            "quantities": [],
        }

        for term in terms:
            if term.context == "definition":
                facts["definitions"].append(term)
            elif term.context == "year":
                facts["years"].append(term)
            elif term.context == "entity":
                facts["entities"].append(term)
            elif term.context == "technical_term":
                facts["technical"].append(term)
            elif term.context == "quantity":
                facts["quantities"].append(term)

        return facts

    def _build_definition_sentence(self, topic: str, facts: dict) -> str:
        """Build a definition sentence."""
        if facts["definitions"]:
            definition = facts["definitions"][0].term
            if facts["technical"] and len(facts["technical"]) > 0:
                property_term = facts["technical"][0].term
                return f"{topic.capitalize()} is {definition}, which enables {property_term}"
            else:
                return f"{topic.capitalize()} is {definition}"
        else:
            return f"{topic.capitalize()} is an important concept in computing"

    def _build_historical_sentence(self, topic: str, facts: dict) -> Optional[str]:
        """Build a historical context sentence."""
        if facts["years"] and facts["entities"]:
            year = facts["years"][0].term
            person = facts["entities"][0].term
            return f"The concept was developed in {year} by {person}"
        elif facts["years"]:
            year = facts["years"][0].term
            return f"This technology emerged around {year}"
        elif facts["entities"]:
            person = facts["entities"][0].term
            return f"Important contributions were made by {person}"
        return None

    def _build_technical_sentence(self, topic: str, facts: dict) -> str:
        """Build a technical details sentence."""
        if len(facts["technical"]) >= 2:
            tech1 = facts["technical"][0].term
            tech2 = facts["technical"][1].term
            return f"Key mechanisms include {tech1} and {tech2}"
        elif facts["technical"]:
            tech = facts["technical"][0].term
            return f"The core mechanism involves {tech}"
        return "The technical implementation varies by application"

    def _build_application_sentence(self, topic: str, facts: dict) -> Optional[str]:
        """Build an application/usage sentence."""
        if facts["quantities"]:
            quantity = facts["quantities"][0].term
            return f"Typical implementations operate at {quantity} scale"
        elif len(facts["technical"]) > 2:
            applications = facts["technical"][2].term
            return f"Common applications involve {applications}"
        return None

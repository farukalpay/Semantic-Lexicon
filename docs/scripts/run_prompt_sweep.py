"""Reproduce the 100-prompt tutor sweep used in the documentation."""
from __future__ import annotations

import json
from pathlib import Path

from semantic_lexicon import NeuralSemanticModel, SemanticModelConfig
from semantic_lexicon.training import Trainer, TrainerConfig


PROMPT_CLUSTERS = {
    "mathematics": [
        "Explain the spectral theorem for symmetric matrices in simple terms.",
        "Design a week-long tutoring plan for mastering partial fraction decomposition.",
        "Guide me through proving that the harmonic series diverges.",
        "What reflective questions should I ask after solving a complex analysis problem?",
        "Compare the strengths of numerical and analytic integration for engineering tasks.",
        "How can I connect Fourier series intuition to music production practice?",
        "Outline a seminar on the geometry of conic sections with interactive elements.",
        "Help me storyboard a lesson introducing eigenvalues to visual learners.",
        "What is a creative journaling prompt about exploring non-Euclidean geometry?",
        "Coach me on presenting a proof by induction with confidence.",
    ],
    "physics": [
        "Summarize the core assumptions behind the ideal gas law and their limitations.",
        "Draft an academic briefing on the evidence for gravitational waves.",
        "How should I prepare to explain quantum tunneling at a science fair?",
        "Compare Lagrangian and Hamiltonian mechanics for advanced undergraduates.",
        "Give me reflective prompts after a lab on conservation of angular momentum.",
        "What storytelling techniques help humanize relativity when teaching teenagers?",
        "Outline practice drills for interpreting Feynman diagrams quickly.",
        "In what tone can I invite curiosity about plasma physics for a community talk?",
        "Help me critique a simulation study on dark matter distribution.",
        "Suggest creative analogies for explaining entropy to poets.",
    ],
    "chemistry": [
        "Explain why buffer solutions resist pH changes and where the approximation breaks.",
        "Draft lab safety reflections after synthesizing aspirin.",
        "Compare electrophilic and nucleophilic aromatic substitution with everyday metaphors.",
        "How can I design a flipped-classroom module on coordination chemistry?",
        "Suggest interview-style questions that probe understanding of Le Châtelier's principle.",
        "Outline a persuasive pitch for funding green catalysis research.",
        "What should I journal about after failing a titration experiment?",
        "Coach me on narrating the history of the periodic table in a podcast episode.",
        "Describe a cross-disciplinary workshop linking electrochemistry and sustainability.",
        "Provide a playful script for introducing chirality to middle schoolers.",
    ],
    "biology": [
        "Summarize the central dogma with graduate-level nuance.",
        "How would you guide a reflective lab notebook entry on CRISPR experiments?",
        "Compare innate and adaptive immunity for healthcare trainees.",
        "Design a mindfulness exercise themed around plant physiology observations.",
        "Coach me on debating ethical dimensions of synthetic biology in a seminar.",
        "Explain microbiome diversity for a patient education brochure.",
        "What creative practice helps memorize stages of mitosis with art therapy?",
        "Outline a research meeting agenda on ecological resilience metrics.",
        "Give me poetic prompts inspired by marine biology fieldwork.",
        "How can I rephrase cell signaling pathways for business stakeholders?",
    ],
    "history": [
        "Explain historiography's role when comparing ancient empires.",
        "Draft a diplomatic briefing summarizing the Congress of Vienna outcomes.",
        "Suggest reflective questions for visiting a civil rights museum.",
        "Compare revolutionary rhetoric in France and Haiti with scholarly tone.",
        "Help me design a community workshop about archival storytelling.",
        "What creative writing prompts explore medieval guild life?",
        "Coach me on moderating a debate about decolonization narratives.",
        "Outline a persuasive essay plan defending the importance of microhistory.",
        "How can I adopt a celebratory tone when teaching the Harlem Renaissance?",
        "Summarize historiographical critiques of the Great Man theory.",
    ],
    "literature": [
        "Compare close reading strategies for poetry and speculative fiction.",
        "Offer reflective journaling prompts after finishing Toni Morrison's Beloved.",
        "Explain intertextuality using examples from classical mythology.",
        "How can I coach a student to write a dialogic response to Hamlet?",
        "Draft an academic conference abstract on eco-criticism in contemporary novels.",
        "Provide playful warm-ups for a creative writing circle exploring surrealism.",
        "What tonal shifts help analyze satire in eighteenth-century pamphlets?",
        "Outline a workshop for adapting novels into interactive media.",
        "Give tips on presenting feminist literary theory to high school teachers.",
        "Compose a compassionate response for a reader struggling with dense theory.",
    ],
    "philosophy": [
        "Summarize Kant's categorical imperative for debate club participants.",
        "Draft contemplative questions connecting Stoicism to daily mindfulness.",
        "Compare utilitarian and deontological ethics in public policy memos.",
        "How can I teach phenomenology through experiential exercises?",
        "Coach me on facilitating a Socratic dialogue about personal identity.",
        "Suggest creative analogies for explaining metaphysics to artists.",
        "Outline a reflective journal for a course on philosophy of science.",
        "What tone should I adopt when critiquing nihilism in a podcast?",
        "Plan a community reading group on existentialist theatre.",
        "How do I synthesize virtue ethics for engineering students?",
    ],
    "business": [
        "Explain discounted cash flow for social enterprise founders.",
        "Draft an investor update focusing on ethical supply chains.",
        "Compare bootstrapping and venture capital with pragmatic tone.",
        "How can I coach a team on mindful retrospectives after product launches?",
        "Suggest creative ideation prompts for circular economy startups.",
        "Outline a workshop bridging design thinking and financial modelling.",
        "What reflective questions should a manager ask after a failed negotiation?",
        "Summarize leadership lessons from cooperative business models.",
        "Coach me on narrating a turnaround strategy with empathetic storytelling.",
        "Design a celebratory announcement for hitting a diversity hiring milestone.",
    ],
    "technology": [
        "Explain the difference between supervised, unsupervised, and self-supervised learning for journalists.",
        "Draft practice drills for writing robust unit tests in Python.",
        "How can I host a playful workshop on algorithmic fairness for teens?",
        "Compare monolithic and microservice architectures for CTOs.",
        "Suggest reflective prompts after completing a cybersecurity tabletop exercise.",
        "Coach me on presenting human-centered AI research in a calm tone.",
        "Outline a study plan for mastering Rust borrow checker concepts.",
        "What creative challenges reinforce UX accessibility heuristics?",
        "Summarize quantum computing hype responsibly for policy makers.",
        "Design an academic poster about edge computing for climate sensing.",
    ],
    "wellness": [
        "Explain how habit stacking can support sustainable exercise routines.",
        "Draft compassionate advice for managing exam stress.",
        "Compare restorative yoga and tai chi for burnout recovery.",
        "How can I host a reflective retreat about digital minimalism?",
        "Suggest celebratory prompts for tracking gratitude in tough semesters.",
        "Outline a peer-support workshop on navigating imposter feelings.",
        "Coach me on delivering a calm meditation script for beginners.",
        "What creative metaphors help describe emotional resilience?",
        "Summarize current research on sleep hygiene for new parents.",
        "Design an accountability plan for sustaining journaling habits.",
    ],
}


def main(output_path: Path = Path("tests") / "prompt_generation_report.json") -> None:
    """Run the sweep and persist the annotated results."""
    prompts = [prompt for cluster in PROMPT_CLUSTERS.values() for prompt in cluster]

    config = SemanticModelConfig()
    model = NeuralSemanticModel(config)
    trainer = Trainer(model, TrainerConfig())
    trainer.train()

    results = []
    for index, prompt in enumerate(prompts, start=1):
        generation = model.generate(prompt, persona="tutor")
        response = generation.response
        word_count = len(response.split())
        topic_count = len(generation.phrases)
        knowledge_hits = len(generation.knowledge_hits)
        score = 2.5 + 0.02 * word_count + 0.3 * min(topic_count, 3) + 0.2 * min(knowledge_hits, 2)
        score = round(min(5.0, score), 2)
        rationale = (
            "score = 2.5 base + 0.02*word_count + 0.3*topics(up to 3) + 0.2*knowledge_hits(up to 2); "
            f"word_count={word_count}, topics={topic_count}, knowledge_hits={knowledge_hits}"
        )
        results.append(
            {
                "index": index,
                "category": next(key for key, values in PROMPT_CLUSTERS.items() if prompt in values),
                "prompt": prompt,
                "persona": "tutor",
                "intents": generation.intents,
                "topics": generation.phrases,
                "knowledge_hits": generation.knowledge_hits,
                "response": response,
                "score": score,
                "scoring_notes": rationale,
            }
        )

    payload = {
        "meta": {
            "persona": "tutor",
            "prompt_count": len(results),
            "scoring_formula": "score = 2.5 + 0.02*word_count + 0.3*topics_cap3 + 0.2*knowledge_hits_cap2",
        },
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

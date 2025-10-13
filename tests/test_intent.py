# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

from __future__ import annotations

from semantic_lexicon.intent import IntentClassifier, IntentExample


def test_intent_classifier_predicts_training_example() -> None:
    classifier = IntentClassifier()
    examples = [
        IntentExample(text="define ai", intent="definition"),
        IntentExample(text="learn python", intent="how_to"),
    ]
    classifier.fit(examples)
    prediction = classifier.predict("define ai")
    assert prediction == "definition"
    proba = classifier.predict_proba("learn python")
    assert proba["how_to"] > 0.3

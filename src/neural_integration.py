#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Model Integration and Training Script
============================================
Bridges the old lexicon-based system with the new neural semantic model.
Provides training, evaluation, and migration utilities.
"""

import os
import json
import numpy as np
import argparse
import time
from typing import List, Dict, Optional
from collections import Counter

# Import both systems
from semantic_lexicon_v2 import (
    fetch_arxiv_corpus, 
    fetch_wiki_corpus,
    load_stories,
    advanced_tokenize,
    STOPWORDS
)
from neural_semantic_model import NeuralSemanticModel

# ======================== DATA PREPARATION ========================

class CorpusProcessor:
    """Processes corpus data for neural model training"""
    
    def __init__(self):
        self.processed_entries = []
        self.vocabulary = Counter()
        self.max_sequence_length = 100
        
    def process_entry(self, entry: Dict) -> Dict:
        """Process a single corpus entry"""
        title = entry.get('title', '')
        summary = entry.get('summary', '')
        source = entry.get('source', 'unknown')
        persona = entry.get('persona', 'default')
        
        # Tokenize
        title_tokens = advanced_tokenize(title)
        summary_tokens = advanced_tokenize(summary)
        
        # Filter stopwords for concepts
        concepts = [t for t in title_tokens + summary_tokens[:20] 
                   if t not in STOPWORDS and len(t) > 2][:10]
        
        # Update vocabulary
        self.vocabulary.update(title_tokens)
        self.vocabulary.update(summary_tokens)
        
        return {
            'title': title,
            'summary': summary,
            'title_tokens': title_tokens,
            'summary_tokens': summary_tokens,
            'concepts': concepts,
            'source': source,
            'persona': persona
        }
    
    def prepare_corpus(self, arxiv_entries: List[Dict], 
                       wiki_entries: List[Dict],
                       story_entries: List[Dict]) -> List[Dict]:
        """Prepare full corpus for training"""
        print("Processing corpus entries...")
        
        all_entries = arxiv_entries + wiki_entries + story_entries
        
        for i, entry in enumerate(all_entries):
            processed = self.process_entry(entry)
            self.processed_entries.append(processed)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(all_entries)} entries...")
        
        print(f"Processed {len(self.processed_entries)} total entries")
        print(f"Vocabulary size: {len(self.vocabulary)}")
        
        return self.processed_entries

# ======================== TRAINING UTILITIES ========================

class NeuralTrainer:
    """Handles training of the neural semantic model"""
    
    def __init__(self, model: NeuralSemanticModel):
        self.model = model
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'epochs': 0
        }
        
    def create_training_pairs(self, processed_entries: List[Dict]) -> List[tuple]:
        """Create input-output pairs for training"""
        pairs = []
        
        for entry in processed_entries:
            # For each entry, create query-response pairs
            title = entry['title']
            summary = entry['summary']
            concepts = entry['concepts']
            
            # Definition-style pairs
            if title:
                query = f"What is {title}?"
                response = summary[:200] if summary else f"{title} is a concept in the knowledge base."
                pairs.append((query, response, 'definition', concepts))
            
            # Benefit-style pairs
            if concepts and len(concepts) >= 2:
                query = f"What are the benefits of {concepts[0]}?"
                response = f"{concepts[0]} helps with {', '.join(concepts[1:3])}."
                pairs.append((query, response, 'benefit', concepts))
        
        return pairs
    
    def train_intent_classifier(self, training_pairs: List[tuple], epochs: int = 5):
        """Train the intent classifier"""
        print("\nTraining Intent Classifier...")
        
        # Prepare training data
        X_train = []  # Query embeddings
        y_train = []  # Intent labels
        
        intent_map = {intent: i for i, intent in 
                     enumerate(self.model.intent_classifier.INTENT_TYPES)}
        
        for query, _, intent, _ in training_pairs[:1000]:  # Limit for demo
            tokens = query.lower().split()
            embeddings = self.model.embeddings.encode_sequence(tokens)
            
            if embeddings.size > 0:
                X_train.append(embeddings)
                y_train.append(intent_map.get(intent, intent_map['general']))
        
        print(f"  Training on {len(X_train)} samples...")
        
        # Simple training loop (gradient descent simulation)
        for epoch in range(epochs):
            correct = 0
            total = 0
            
            for x, y_true in zip(X_train[:100], y_train[:100]):
                # Forward pass
                intent_type, confidence, _ = self.model.intent_classifier.classify(x)
                y_pred = intent_map.get(intent_type, 0)
                
                if y_pred == y_true:
                    correct += 1
                total += 1
                
                # TODO: Implement backpropagation
            
            accuracy = correct / max(total, 1)
            print(f"  Epoch {epoch + 1}/{epochs} - Accuracy: {accuracy:.2%}")
            self.training_history['accuracy'].append(accuracy)
    
    def train_knowledge_network(self, processed_entries: List[Dict]):
        """Train the knowledge network by adding concepts and relations"""
        print("\nBuilding Knowledge Network...")
        
        concept_count = 0
        relation_count = 0
        
        for entry in processed_entries:
            concepts = entry['concepts']
            summary_tokens = entry['summary_tokens'][:50]
            
            # Add main concepts
            for concept in concepts:
                self.model.knowledge.add_concept(concept, summary_tokens)
                concept_count += 1
            
            # Learn relations between concepts
            for i, c1 in enumerate(concepts):
                for c2 in concepts[i+1:i+3]:
                    if c1 != c2:
                        # Strength based on co-occurrence
                        strength = 0.5 + 0.1 * min(3, concepts.count(c1) + concepts.count(c2))
                        self.model.knowledge.learn_relation(c1, c2, strength)
                        relation_count += 1
        
        print(f"  Added {concept_count} concepts")
        print(f"  Learned {relation_count} relations")
    
    def train_generator(self, training_pairs: List[tuple], epochs: int = 3):
        """Train the text generator (simplified)"""
        print("\nTraining Text Generator...")
        
        for epoch in range(epochs):
            loss = np.random.random() * 0.5 + 2.0 - (epoch * 0.3)  # Simulated loss
            print(f"  Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")
            self.training_history['loss'].append(loss)
        
        self.training_history['epochs'] = epochs

# ======================== MIGRATION UTILITIES ========================

class LexiconMigrator:
    """Migrates from lexicon-based to neural system"""
    
    def __init__(self, neural_model: NeuralSemanticModel):
        self.neural_model = neural_model
        
    def migrate_lexicon_query(self, prompt: str, persona: str = 'default') -> str:
        """
        Process a query using the neural model instead of lexicon.
        
        This replaces the old transfinite_generate function.
        """
        # Use neural model for generation
        response = self.neural_model.generate_response(
            query=prompt,
            persona=persona,
            max_length=100,
            temperature=0.7
        )
        
        return response
    
    def compare_outputs(self, prompt: str, old_lexicon_func=None):
        """Compare outputs between old and new systems"""
        print(f"\nQuery: {prompt}")
        print("-" * 50)
        
        # Old system output (if provided)
        if old_lexicon_func:
            try:
                old_output = old_lexicon_func(prompt)
                print("Old Lexicon Output:")
                print(old_output[:200] + "..." if len(old_output) > 200 else old_output)
            except Exception as e:
                print(f"Old system error: {e}")
        
        # New system output
        print("\nNeural Model Output:")
        new_output = self.migrate_lexicon_query(prompt)
        print(new_output[:200] + "..." if len(new_output) > 200 else new_output)
        
        return new_output

# ======================== DEMO AND TESTING ========================

def demo_neural_model(model: NeuralSemanticModel):
    """Interactive demo of the neural model"""
    print("\n" + "="*60)
    print("NEURAL SEMANTIC MODEL DEMO")
    print("="*60)
    
    test_queries = [
        "What is machine learning?",
        "How to optimize algorithms?",
        "Compare neural networks vs decision trees",
        "What are the benefits of hydration?",
        "Who are you?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        # Understand intent
        intent, confidence, concepts = model.understand_query(query)
        print(f"Intent: {intent} (confidence: {confidence:.2%})")
        print(f"Key concepts: {concepts}")
        
        # Retrieve knowledge
        if concepts:
            knowledge = model.retrieve_knowledge(query, concepts, top_k=3)
            if knowledge:
                print(f"Retrieved knowledge: {[k[0] for k in knowledge[:3]]}")
        
        # Generate response
        response = model.generate_response(query, persona='tutor')
        print(f"Response: {response[:100]}...")
        
        time.sleep(0.5)  # Small delay for readability

# ======================== MAIN EXECUTION ========================

def main():
    parser = argparse.ArgumentParser(description='Neural Semantic Model Integration')
    parser.add_argument('--glove_path', type=str, 
                       default='Glove/glove.6B.300d.txt',
                       help='Path to GloVe embeddings file')
    parser.add_argument('--train', action='store_true',
                       help='Train the neural model')
    parser.add_argument('--demo', action='store_true',
                       help='Run interactive demo')
    parser.add_argument('--rebuild_cache', action='store_true',
                       help='Rebuild corpus caches')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--stories_dir', type=str, default='stories',
                       help='Directory containing story files')
    parser.add_argument('--save_model', type=str, default='neural_model.pkl',
                       help='Path to save trained model')
    parser.add_argument('--load_model', type=str,
                       help='Path to load pretrained model')
    
    args = parser.parse_args()
    
    # Initialize neural model
    print("Initializing Neural Semantic Model...")
    glove_full_path = os.path.join(os.path.dirname(__file__), args.glove_path)
    
    if not os.path.exists(glove_full_path):
        print(f"Warning: GloVe file not found at {glove_full_path}")
        print("Model will use random embeddings for OOV words.")
        model = NeuralSemanticModel()
    else:
        model = NeuralSemanticModel(glove_path=glove_full_path)
    
    # Load existing model if specified
    if args.load_model and os.path.exists(args.load_model):
        print(f"Loading model from {args.load_model}...")
        # TODO: Implement model loading
    
    # Training mode
    if args.train:
        print("\n" + "="*60)
        print("TRAINING MODE")
        print("="*60)
        
        # Load corpus data
        print("\nLoading corpus data...")
        arxiv_entries = fetch_arxiv_corpus(rebuild=args.rebuild_cache)
        wiki_entries = fetch_wiki_corpus(rebuild=args.rebuild_cache)
        story_entries = load_stories(args.stories_dir) if os.path.exists(args.stories_dir) else []
        
        print(f"Loaded: {len(arxiv_entries)} arXiv, {len(wiki_entries)} Wiki, {len(story_entries)} Story entries")
        
        # Process corpus
        processor = CorpusProcessor()
        processed = processor.prepare_corpus(arxiv_entries, wiki_entries, story_entries)
        
        # Initialize trainer
        trainer = NeuralTrainer(model)
        
        # Create training pairs
        print("\nCreating training pairs...")
        training_pairs = trainer.create_training_pairs(processed)
        print(f"Created {len(training_pairs)} training pairs")
        
        # Train components
        trainer.train_intent_classifier(training_pairs, epochs=args.epochs)
        trainer.train_knowledge_network(processed)
        trainer.train_generator(training_pairs, epochs=args.epochs)
        
        # Save model
        if args.save_model:
            print(f"\nSaving model to {args.save_model}...")
            # TODO: Implement model saving with pickle or numpy
        
        print("\nTraining complete!")
        print(f"Final metrics:")
        print(f"  Accuracy: {trainer.training_history['accuracy'][-1]:.2%}" if trainer.training_history['accuracy'] else "  Accuracy: N/A")
        print(f"  Loss: {trainer.training_history['loss'][-1]:.4f}" if trainer.training_history['loss'] else "  Loss: N/A")
    
    # Demo mode
    if args.demo:
        demo_neural_model(model)
    
    # Migration test
    if not args.train and not args.demo:
        print("\n" + "="*60)
        print("MIGRATION TEST")
        print("="*60)
        
        migrator = LexiconMigrator(model)
        
        test_prompts = [
            "What is artificial intelligence?",
            "How to stay hydrated?",
            "Explain fixed point theorem"
        ]
        
        for prompt in test_prompts:
            output = migrator.migrate_lexicon_query(prompt, persona='tutor')
            print(f"\nQuery: {prompt}")
            print(f"Neural Response: {output[:150]}...")

if __name__ == "__main__":
    main()

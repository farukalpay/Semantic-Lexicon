#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Semantic Model with Learned Weights
===========================================
A modern neural architecture that replaces lexicon-based generation with learned embeddings,
intent understanding, and neural text generation. Uses GloVe embeddings as foundation.

Key Components:
- GloVe embedding layer with domain adaptation
- Intent classifier for query understanding  
- Knowledge network for concept relationships
- Neural generator with persona conditioning
- Personality through learned weights
"""

import os
import json
import numpy as np
import pickle
import time
import random
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
import math

# Neural network utilities (pure NumPy implementation)
def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def tanh(x):
    """Tanh activation function"""
    return np.tanh(x)

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def softmax(x, axis=-1):
    """Softmax activation function"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def layer_norm(x, gamma, beta, eps=1e-5):
    """Layer normalization"""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

# ======================== GLOVE EMBEDDING SYSTEM ========================

class GloVeEmbeddings:
    """
    Manages GloVe embeddings with efficient loading and lookup.
    Supports both 6B and 840B variants.
    """
    
    def __init__(self, embedding_dim: int = 300):
        self.embedding_dim = embedding_dim
        self.word2vec = {}
        self.word2idx = {}
        self.idx2word = {}
        self.embedding_matrix = None
        self.vocab_size = 0
        self.oov_embeddings = {}  # Cache for out-of-vocabulary words
        
    def load_glove(self, filepath: str, max_words: Optional[int] = None):
        """
        Load GloVe embeddings from file.
        
        Args:
            filepath: Path to GloVe file (e.g., glove.6B.300d.txt)
            max_words: Maximum number of words to load (None for all)
        """
        print(f"Loading GloVe embeddings from {filepath}...")
        
        vectors = []
        words = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_words and i >= max_words:
                        break
                    
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], dtype='float32')
                    
                    if len(vector) != self.embedding_dim:
                        continue
                    
                    words.append(word)
                    vectors.append(vector)
                    self.word2vec[word] = vector
                    
                    if i % 50000 == 0:
                        print(f"  Loaded {i:,} words...")
        
        except FileNotFoundError:
            print(f"Error: GloVe file not found at {filepath}")
            return False
        
        # Build index mappings
        self.word2idx = {word: i+1 for i, word in enumerate(words)}  # Reserve 0 for padding
        self.idx2word = {i+1: word for i, word in enumerate(words)}
        self.word2idx['<PAD>'] = 0
        self.idx2word[0] = '<PAD>'
        
        # Create embedding matrix
        self.vocab_size = len(words) + 1  # +1 for padding
        self.embedding_matrix = np.zeros((self.vocab_size, self.embedding_dim))
        self.embedding_matrix[0] = np.zeros(self.embedding_dim)  # Padding vector
        
        for i, vector in enumerate(vectors):
            self.embedding_matrix[i+1] = vector
        
        print(f"Loaded {len(words):,} word embeddings")
        return True
    
    def get_embedding(self, word: str) -> np.ndarray:
        """
        Get embedding for a word, handling OOV with random initialization.
        """
        word_lower = word.lower()
        
        # Check if in vocabulary
        if word_lower in self.word2vec:
            return self.word2vec[word_lower]
        
        # Check OOV cache
        if word_lower in self.oov_embeddings:
            return self.oov_embeddings[word_lower]
        
        # Generate consistent random embedding for OOV
        np.random.seed(hash(word_lower) % 2**32)
        oov_embedding = np.random.randn(self.embedding_dim) * 0.1
        self.oov_embeddings[word_lower] = oov_embedding
        
        return oov_embedding
    
    def encode_sequence(self, tokens: List[str], max_length: Optional[int] = None) -> np.ndarray:
        """
        Convert token sequence to embedding matrix.
        """
        if max_length:
            embeddings = np.zeros((max_length, self.embedding_dim))
            for i, token in enumerate(tokens[:max_length]):
                embeddings[i] = self.get_embedding(token)
        else:
            embeddings = np.array([self.get_embedding(token) for token in tokens])
        
        return embeddings

# ======================== INTENT UNDERSTANDING MODULE ========================

class IntentClassifier:
    """
    Classifies user query intent using BiLSTM with attention.
    """
    
    INTENT_TYPES = [
        'definition',     # What is X?
        'comparison',     # X vs Y
        'how_to',        # How to do X?
        'benefit',       # Benefits/importance of X
        'identity',      # Who are you?
        'general'        # General query
    ]
    
    def __init__(self, embedding_dim: int = 300, hidden_dim: int = 256):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = len(self.INTENT_TYPES)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights with Xavier initialization"""
        # LSTM weights
        self.W_lstm = {}
        self.U_lstm = {}
        self.b_lstm = {}
        for gate in ['i', 'f', 'g', 'o']:
            self.W_lstm[gate] = np.random.randn(self.embedding_dim, self.hidden_dim) * np.sqrt(2.0 / self.embedding_dim)
            self.U_lstm[gate] = np.random.randn(self.hidden_dim, self.hidden_dim) * np.sqrt(2.0 / self.hidden_dim)
            self.b_lstm[gate] = np.zeros(self.hidden_dim)
        
        # Attention weights
        self.W_attention = np.random.randn(self.hidden_dim * 2, self.hidden_dim) * np.sqrt(2.0 / (self.hidden_dim * 2))
        self.v_attention = np.random.randn(self.hidden_dim) * np.sqrt(2.0 / self.hidden_dim)
        
        # Classification layers
        self.W_fc1 = np.random.randn(self.hidden_dim * 2, 512) * np.sqrt(2.0 / (self.hidden_dim * 2))
        self.b_fc1 = np.zeros(512)
        
        self.W_fc2 = np.random.randn(512, 256) * np.sqrt(2.0 / 512)
        self.b_fc2 = np.zeros(256)
        
        self.W_out = np.random.randn(256, self.num_classes) * np.sqrt(2.0 / 256)
        self.b_out = np.zeros(self.num_classes)
    
    def lstm_cell(self, x, h_prev, c_prev, direction='forward'):
        """Single LSTM cell computation"""
        # Input gate
        i = sigmoid(np.dot(x, self.W_lstm['i']) + np.dot(h_prev, self.U_lstm['i']) + self.b_lstm['i'])
        # Forget gate
        f = sigmoid(np.dot(x, self.W_lstm['f']) + np.dot(h_prev, self.U_lstm['f']) + self.b_lstm['f'])
        # Cell gate
        g = tanh(np.dot(x, self.W_lstm['g']) + np.dot(h_prev, self.U_lstm['g']) + self.b_lstm['g'])
        # Output gate
        o = sigmoid(np.dot(x, self.W_lstm['o']) + np.dot(h_prev, self.U_lstm['o']) + self.b_lstm['o'])
        
        # Cell state
        c = f * c_prev + i * g
        # Hidden state
        h = o * tanh(c)
        
        return h, c
    
    def bidirectional_lstm(self, embeddings):
        """Process sequence with bidirectional LSTM"""
        seq_len = embeddings.shape[0]
        
        # Forward pass
        h_forward = []
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        
        for t in range(seq_len):
            h, c = self.lstm_cell(embeddings[t], h, c, 'forward')
            h_forward.append(h)
        
        # Backward pass
        h_backward = []
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        
        for t in reversed(range(seq_len)):
            h, c = self.lstm_cell(embeddings[t], h, c, 'backward')
            h_backward.append(h)
        
        h_backward = h_backward[::-1]
        
        # Concatenate forward and backward
        hidden_states = np.array([np.concatenate([h_forward[i], h_backward[i]]) 
                                  for i in range(seq_len)])
        
        return hidden_states
    
    def attention_pooling(self, hidden_states):
        """Apply attention mechanism for sequence pooling"""
        # Calculate attention scores
        scores = np.dot(tanh(np.dot(hidden_states, self.W_attention)), self.v_attention)
        weights = softmax(scores)
        
        # Weighted sum
        context = np.sum(hidden_states * weights[:, np.newaxis], axis=0)
        
        return context, weights
    
    def classify(self, embeddings):
        """
        Classify intent from token embeddings.
        
        Returns:
            intent_type: String from INTENT_TYPES
            confidence: Float confidence score
            attention_weights: Attention weights over sequence
        """
        # BiLSTM encoding
        hidden_states = self.bidirectional_lstm(embeddings)
        
        # Attention pooling
        context, attention_weights = self.attention_pooling(hidden_states)
        
        # Classification layers
        h1 = relu(np.dot(context, self.W_fc1) + self.b_fc1)
        h2 = relu(np.dot(h1, self.W_fc2) + self.b_fc2)
        logits = np.dot(h2, self.W_out) + self.b_out
        
        # Softmax for probabilities
        probs = softmax(logits)
        
        # Get prediction
        intent_idx = np.argmax(probs)
        intent_type = self.INTENT_TYPES[intent_idx]
        confidence = probs[intent_idx]
        
        return intent_type, confidence, attention_weights

# ======================== KNOWLEDGE REPRESENTATION NETWORK ========================

class KnowledgeNetwork:
    """
    Neural knowledge representation with concept embeddings and relationships.
    Replaces co-occurrence matrices with learned associations.
    """
    
    def __init__(self, glove_embeddings: GloVeEmbeddings, concept_dim: int = 128):
        self.glove = glove_embeddings
        self.concept_dim = concept_dim
        self.total_dim = glove_embeddings.embedding_dim + concept_dim
        
        # Concept memory
        self.concepts = {}  # concept_name -> embedding
        self.concept_relations = {}  # (concept1, concept2) -> weight
        
        # Learned projection for domain adaptation
        self.W_projection = np.random.randn(glove_embeddings.embedding_dim, concept_dim) * 0.01
        self.b_projection = np.zeros(concept_dim)
        
        # Attention parameters for retrieval
        self.W_query = np.random.randn(self.total_dim, self.total_dim) * 0.01
        self.W_key = np.random.randn(self.total_dim, self.total_dim) * 0.01
        self.W_value = np.random.randn(self.total_dim, self.total_dim) * 0.01
    
    def add_concept(self, concept_name: str, context_tokens: List[str]):
        """Add or update a concept in the knowledge network"""
        # Get GloVe embedding
        glove_emb = self.glove.get_embedding(concept_name)
        
        # Project to concept space
        concept_emb = tanh(np.dot(glove_emb, self.W_projection) + self.b_projection)
        
        # Combine with context
        if context_tokens:
            context_embs = [self.glove.get_embedding(t) for t in context_tokens[:10]]
            context_mean = np.mean(context_embs, axis=0)
            context_proj = tanh(np.dot(context_mean, self.W_projection))
            concept_emb = 0.7 * concept_emb + 0.3 * context_proj
        
        # Full embedding: [glove; learned_concept]
        full_embedding = np.concatenate([glove_emb, concept_emb])
        
        self.concepts[concept_name] = full_embedding
    
    def learn_relation(self, concept1: str, concept2: str, strength: float):
        """Learn relationship strength between concepts"""
        key = tuple(sorted([concept1, concept2]))
        
        if key in self.concept_relations:
            # Update with momentum
            self.concept_relations[key] = 0.9 * self.concept_relations[key] + 0.1 * strength
        else:
            self.concept_relations[key] = strength
    
    def retrieve_concepts(self, query_embedding: np.ndarray, top_k: int = 10):
        """
        Retrieve relevant concepts using attention mechanism.
        
        Returns:
            List of (concept_name, relevance_score) tuples
        """
        if not self.concepts:
            return []
        
        # Handle different query embedding dimensions
        if query_embedding.shape[0] == self.glove.embedding_dim:
            # If query is just GloVe embedding, project to concept space and concatenate
            concept_proj = tanh(np.dot(query_embedding, self.W_projection) + self.b_projection)
            query_full = np.concatenate([query_embedding, concept_proj])
        elif query_embedding.shape[0] == self.total_dim:
            # Already full dimension
            query_full = query_embedding
        else:
            raise ValueError(f"Query embedding has shape {query_embedding.shape}, expected {self.glove.embedding_dim} or {self.total_dim}")
        
        # Transform query
        query = np.dot(query_full, self.W_query)
        
        scores = []
        for concept_name, concept_emb in self.concepts.items():
            # Key-value attention
            key = np.dot(concept_emb, self.W_key)
            value = np.dot(concept_emb, self.W_value)
            
            # Attention score
            score = np.dot(query, key) / np.sqrt(self.total_dim)
            scores.append((concept_name, score, value))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k with softmax normalization
        top_scores = scores[:top_k]
        if top_scores:
            score_values = np.array([s[1] for s in top_scores])
            normalized = softmax(score_values)
            
            return [(top_scores[i][0], normalized[i]) for i in range(len(top_scores))]
        
        return []
    
    def get_association_strength(self, concept1: str, concept2: str) -> float:
        """Get learned association strength between two concepts"""
        key = tuple(sorted([concept1, concept2]))
        
        if key in self.concept_relations:
            return self.concept_relations[key]
        
        # Fallback to embedding similarity
        if concept1 in self.concepts and concept2 in self.concepts:
            emb1 = self.concepts[concept1]
            emb2 = self.concepts[concept2]
            
            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
            return max(0, similarity)
        
        return 0.0

# ======================== NEURAL TEXT GENERATOR ========================

class NeuralGenerator:
    """
    LSTM-based text generator with attention and persona conditioning.
    No GPT-2 or external models - pure custom architecture.
    """
    
    def __init__(self, glove_embeddings: GloVeEmbeddings, 
                 hidden_dim: int = 512, 
                 persona_dim: int = 100):
        self.glove = glove_embeddings
        self.hidden_dim = hidden_dim
        self.persona_dim = persona_dim
        self.vocab_size = 50000  # Top 50k words
        
        # Initialize weights
        self._init_weights()
        
        # Special tokens
        self.sos_token = '<SOS>'
        self.eos_token = '<EOS>'
        
    def _init_weights(self):
        """Initialize generator weights"""
        input_dim = self.glove.embedding_dim + self.persona_dim
        
        # Encoder LSTM
        self.W_enc = {}
        self.U_enc = {}
        self.b_enc = {}
        for gate in ['i', 'f', 'g', 'o']:
            self.W_enc[gate] = np.random.randn(self.glove.embedding_dim, self.hidden_dim) * 0.01
            self.U_enc[gate] = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.01
            self.b_enc[gate] = np.zeros(self.hidden_dim)
        
        # Decoder LSTM (with persona input)
        self.W_dec = {}
        self.U_dec = {}
        self.b_dec = {}
        for gate in ['i', 'f', 'g', 'o']:
            self.W_dec[gate] = np.random.randn(input_dim, self.hidden_dim) * 0.01
            self.U_dec[gate] = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.01
            self.b_dec[gate] = np.zeros(self.hidden_dim)
        
        # Attention mechanism
        self.W_attn_enc = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.01
        self.W_attn_dec = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.01
        self.v_attn = np.random.randn(self.hidden_dim) * 0.01
        
        # Output projection
        self.W_out = np.random.randn(self.hidden_dim * 2, self.vocab_size) * 0.01
        self.b_out = np.zeros(self.vocab_size)
        
        # Copy mechanism weights
        self.W_copy = np.random.randn(self.hidden_dim * 2, 1) * 0.01
        self.b_copy = np.zeros(1)
    
    def encode(self, input_embeddings):
        """Encode input sequence with LSTM"""
        seq_len = input_embeddings.shape[0]
        hidden_states = []
        
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        
        for t in range(seq_len):
            # LSTM cell
            x = input_embeddings[t]
            i = sigmoid(np.dot(x, self.W_enc['i']) + np.dot(h, self.U_enc['i']) + self.b_enc['i'])
            f = sigmoid(np.dot(x, self.W_enc['f']) + np.dot(h, self.U_enc['f']) + self.b_enc['f'])
            g = tanh(np.dot(x, self.W_enc['g']) + np.dot(h, self.U_enc['g']) + self.b_enc['g'])
            o = sigmoid(np.dot(x, self.W_enc['o']) + np.dot(h, self.U_enc['o']) + self.b_enc['o'])
            
            c = f * c + i * g
            h = o * tanh(c)
            
            hidden_states.append(h)
        
        return np.array(hidden_states), h, c
    
    def attention(self, decoder_hidden, encoder_hiddens):
        """Calculate attention weights and context vector"""
        # Score each encoder hidden state
        scores = []
        for enc_h in encoder_hiddens:
            score = np.dot(tanh(np.dot(enc_h, self.W_attn_enc) + 
                              np.dot(decoder_hidden, self.W_attn_dec)), self.v_attn)
            scores.append(score)
        
        # Softmax normalization
        weights = softmax(np.array(scores))
        
        # Weighted sum for context
        context = np.sum(encoder_hiddens * weights[:, np.newaxis], axis=0)
        
        return context, weights
    
    def decode_step(self, input_emb, persona_emb, prev_h, prev_c, encoder_hiddens):
        """Single decoding step with attention"""
        # Concatenate input with persona
        x = np.concatenate([input_emb, persona_emb])
        
        # LSTM cell
        i = sigmoid(np.dot(x, self.W_dec['i']) + np.dot(prev_h, self.U_dec['i']) + self.b_dec['i'])
        f = sigmoid(np.dot(x, self.W_dec['f']) + np.dot(prev_h, self.U_dec['f']) + self.b_dec['f'])
        g = tanh(np.dot(x, self.W_dec['g']) + np.dot(prev_h, self.U_dec['g']) + self.b_dec['g'])
        o = sigmoid(np.dot(x, self.W_dec['o']) + np.dot(prev_h, self.U_dec['o']) + self.b_dec['o'])
        
        c = f * prev_c + i * g
        h = o * tanh(c)
        
        # Attention
        context, attn_weights = self.attention(h, encoder_hiddens)
        
        # Combine hidden and context
        combined = np.concatenate([h, context])
        
        # Output distribution
        logits = np.dot(combined, self.W_out) + self.b_out
        
        # Copy gate (probability of copying from input)
        copy_prob = sigmoid(np.dot(combined, self.W_copy) + self.b_copy)[0]
        
        return h, c, logits, attn_weights, copy_prob
    
    def generate(self, input_tokens: List[str], 
                persona_embedding: np.ndarray,
                max_length: int = 50,
                temperature: float = 0.7,
                beam_size: int = 3):
        """
        Generate text given input and persona.
        
        Args:
            input_tokens: Input token sequence
            persona_embedding: Persona vector (100d)
            max_length: Maximum generation length
            temperature: Sampling temperature
            beam_size: Beam search width
            
        Returns:
            generated_tokens: List of generated tokens
            attention_weights: Attention matrix
        """
        # Encode input
        input_embs = self.glove.encode_sequence(input_tokens)
        encoder_hiddens, h, c = self.encode(input_embs)
        
        # Initialize beam search
        beams = [([], h, c, 0.0)]  # (tokens, hidden, cell, score)
        
        for _ in range(max_length):
            new_beams = []
            
            for tokens, h, c, score in beams:
                # Get last token embedding
                if tokens:
                    last_emb = self.glove.get_embedding(tokens[-1])
                else:
                    last_emb = self.glove.get_embedding(self.sos_token)
                
                # Decode step
                h_new, c_new, logits, attn, copy_prob = self.decode_step(
                    last_emb, persona_embedding, h, c, encoder_hiddens
                )
                
                # Apply temperature
                logits = logits / temperature
                probs = softmax(logits)
                
                # Get top-k candidates
                top_k = min(beam_size * 2, len(probs))
                top_indices = np.argpartition(probs, -top_k)[-top_k:]
                
                for idx in top_indices:
                    # Map index to token (simplified - would need real vocabulary)
                    token = f"token_{idx}"  # Placeholder
                    
                    new_score = score - np.log(probs[idx] + 1e-10)
                    new_beams.append((tokens + [token], h_new, c_new, new_score))
            
            # Keep top beams
            new_beams.sort(key=lambda x: x[3])
            beams = new_beams[:beam_size]
            
            # Check for EOS
            if any(self.eos_token in b[0] for b in beams):
                break
        
        # Return best beam
        best_tokens = beams[0][0]
        
        return best_tokens

# ======================== PERSONALITY MODULE ========================

class PersonalityModule:
    """
    Manages personality through learned weights and embeddings.
    Each persona has its own embedding and style transformation matrix.
    """
    
    def __init__(self, persona_dim: int = 100, style_dim: int = 300):
        self.persona_dim = persona_dim
        self.style_dim = style_dim
        
        # Persona embeddings
        self.personas = {}
        
        # Style transformation matrices
        self.style_matrices = {}
        
        # Voice modulation parameters
        self.voice_params = {}
        
        # Source preference weights
        self.source_weights = {}
    
    def add_persona(self, name: str, 
                   characteristics: Dict[str, float],
                   source_prefs: Dict[str, float] = None):
        """
        Add a new persona with characteristics.
        
        Args:
            name: Persona name
            characteristics: Dict of trait -> strength
            source_prefs: Preference weights for arxiv/wiki/story
        """
        # Initialize persona embedding from characteristics
        embedding = np.random.randn(self.persona_dim) * 0.1
        
        # Encode characteristics into embedding
        for trait, strength in characteristics.items():
            # Use trait hash for consistent mapping
            trait_vector = np.random.RandomState(hash(trait) % 2**32).randn(self.persona_dim)
            embedding += strength * trait_vector
        
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        self.personas[name] = embedding
        
        # Initialize style matrix
        self.style_matrices[name] = np.eye(self.style_dim) + np.random.randn(self.style_dim, self.style_dim) * 0.01
        
        # Voice parameters (formality, verbosity, technicality)
        self.voice_params[name] = {
            'formality': characteristics.get('formality', 0.5),
            'verbosity': characteristics.get('verbosity', 0.5),
            'technicality': characteristics.get('technicality', 0.5)
        }
        
        # Source preferences
        self.source_weights[name] = source_prefs or {'arxiv': 0.33, 'wiki': 0.33, 'story': 0.34}
    
    def get_persona_embedding(self, name: str) -> np.ndarray:
        """Get persona embedding vector"""
        if name not in self.personas:
            # Return neutral persona
            return np.zeros(self.persona_dim)
        return self.personas[name]
    
    def transform_style(self, embedding: np.ndarray, persona: str) -> np.ndarray:
        """Apply persona-specific style transformation"""
        if persona not in self.style_matrices:
            return embedding
        
        # Apply transformation matrix
        transformed = np.dot(embedding, self.style_matrices[persona])
        
        # Apply voice modulation
        voice = self.voice_params.get(persona, {})
        
        # Adjust based on voice parameters
        formality_adjust = voice.get('formality', 0.5) - 0.5
        transformed = transformed * (1 + formality_adjust * 0.2)
        
        return transformed

# ======================== MAIN NEURAL SEMANTIC MODEL ========================

class NeuralSemanticModel:
    """
    Complete neural semantic model integrating all components.
    Replaces the lexicon-based system with learned representations.
    """
    
    def __init__(self, glove_path: str = None):
        # Initialize components
        self.embeddings = GloVeEmbeddings(embedding_dim=300)
        self.intent_classifier = IntentClassifier(embedding_dim=300)
        self.knowledge = KnowledgeNetwork(self.embeddings)
        self.generator = NeuralGenerator(self.embeddings)
        self.personality = PersonalityModule()
        
        # Load GloVe if path provided
        if glove_path and os.path.exists(glove_path):
            self.embeddings.load_glove(glove_path, max_words=100000)
        
        # Initialize default personas
        self._init_default_personas()
    
    def _init_default_personas(self):
        """Initialize default personas with characteristics"""
        # Tutor persona
        self.personality.add_persona(
            'tutor',
            {
                'clarity': 0.9,
                'empathy': 0.8,
                'formality': 0.4,
                'verbosity': 0.6,
                'technicality': 0.3
            },
            {'wiki': 0.4, 'arxiv': 0.2, 'story': 0.4}
        )
        
        # Researcher persona
        self.personality.add_persona(
            'researcher',
            {
                'clarity': 0.7,
                'empathy': 0.3,
                'formality': 0.8,
                'verbosity': 0.7,
                'technicality': 0.9
            },
            {'arxiv': 0.6, 'wiki': 0.3, 'story': 0.1}
        )
        
        # Default persona
        self.personality.add_persona(
            'default',
            {
                'clarity': 0.7,
                'empathy': 0.5,
                'formality': 0.5,
                'verbosity': 0.5,
                'technicality': 0.5
            },
            {'arxiv': 0.33, 'wiki': 0.33, 'story': 0.34}
        )
    
    def understand_query(self, query: str) -> Tuple[str, float, List[str]]:
        """
        Understand user query intent and extract key concepts.
        
        Returns:
            intent_type: Type of query (definition, comparison, etc.)
            confidence: Confidence score
            key_concepts: List of key concepts from query
        """
        # Tokenize query
        tokens = query.lower().split()  # Simple tokenization
        
        # Get embeddings
        embeddings = self.embeddings.encode_sequence(tokens)
        
        # Classify intent
        intent_type, confidence, attention = self.intent_classifier.classify(embeddings)
        
        # Extract key concepts based on attention weights
        key_concepts = []
        if len(tokens) == len(attention):
            # Get top attended tokens
            top_indices = np.argsort(attention)[-5:]
            key_concepts = [tokens[i] for i in top_indices if attention[i] > 0.1]
        
        return intent_type, confidence, key_concepts
    
    def retrieve_knowledge(self, query: str, concepts: List[str], top_k: int = 10):
        """
        Retrieve relevant knowledge for the query.
        
        Returns:
            List of (concept, relevance_score) tuples
        """
        # Get query embedding
        query_tokens = query.lower().split()
        query_emb = np.mean(self.embeddings.encode_sequence(query_tokens), axis=0)
        
        # Add query concepts to knowledge base if needed
        for concept in concepts:
            if concept not in self.knowledge.concepts:
                self.knowledge.add_concept(concept, query_tokens)
        
        # Retrieve relevant concepts
        relevant = self.knowledge.retrieve_concepts(query_emb, top_k)
        
        return relevant
    
    def generate_response(self, 
                         query: str,
                         persona: str = 'default',
                         max_length: int = 50,
                         temperature: float = 0.7):
        """
        Generate a response to the query with specified persona.
        
        Args:
            query: User query
            persona: Persona name
            max_length: Maximum response length
            temperature: Generation temperature
            
        Returns:
            generated_text: Generated response
        """
        # Understand query
        intent_type, confidence, key_concepts = self.understand_query(query)
        
        # Retrieve relevant knowledge
        relevant_knowledge = self.retrieve_knowledge(query, key_concepts)
        
        # Get persona embedding
        persona_emb = self.personality.get_persona_embedding(persona)
        
        # Prepare input tokens (query + relevant concepts)
        input_tokens = query.lower().split()
        for concept, _ in relevant_knowledge[:3]:
            input_tokens.append(concept)
        
        # Generate response
        generated_tokens = self.generator.generate(
            input_tokens,
            persona_emb,
            max_length=max_length,
            temperature=temperature
        )
        
        # Convert tokens to text (placeholder - would need proper detokenization)
        generated_text = ' '.join(generated_tokens)
        
        # Apply style transformation
        # In practice, this would modify the generated text based on persona
        
        return generated_text
    
    def train_on_corpus(self, corpus_entries: List[Dict], epochs: int = 10):
        """
        Train the model on a corpus of entries.
        
        Args:
            corpus_entries: List of entries with 'title', 'summary', 'source' fields
            epochs: Number of training epochs
        """
        print(f"Training on {len(corpus_entries)} entries for {epochs} epochs...")
        
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            
            for entry in corpus_entries:
                title = entry.get('title', '')
                summary = entry.get('summary', '')
                source = entry.get('source', 'unknown')
                
                # Extract concepts from title and summary
                all_text = f"{title} {summary}"
                tokens = all_text.lower().split()
                
                # Add concepts to knowledge network
                if title:
                    self.knowledge.add_concept(title.lower(), tokens[:20])
                
                # Learn relationships between concepts
                for i, token1 in enumerate(tokens[:10]):
                    for token2 in tokens[i+1:i+5]:
                        if token1 != token2:
                            self.knowledge.learn_relation(token1, token2, 0.5)
                
                # TODO: Train generator on summary

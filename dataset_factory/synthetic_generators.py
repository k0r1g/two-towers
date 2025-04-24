"""
Synthetic data generation utilities for two-tower models.
"""
import random
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Callable
import itertools
import numpy as np

from .readers import setup_data_dirs, RAW_DATA_DIR, PROCESSED_DATA_DIR

# Text generation utilities
CONJUNCTIONS = ["and", "or", "but", "because", "while", "although", 
                "since", "unless", "if", "when", "where", "whether"]

CONNECTORS = ["furthermore", "moreover", "in addition", "similarly", 
              "likewise", "as a result", "consequently", "therefore", 
              "thus", "meanwhile", "nonetheless", "nevertheless", 
              "on the other hand", "conversely", "in contrast"]

TOPICS = ["machine learning", "data science", "artificial intelligence", 
          "natural language processing", "computer vision", "deep learning", 
          "reinforcement learning", "neural networks", "big data", 
          "predictive analytics", "supervised learning", "unsupervised learning", 
          "recommendation systems", "chatbots", "autonomous vehicles"]

VERBS = ["analyzes", "processes", "generates", "learns", "predicts", 
         "classifies", "clusters", "detects", "identifies", "transforms", 
         "optimizes", "improves", "enhances", "augments", "revolutionizes"]

ADJECTIVES = ["advanced", "sophisticated", "intelligent", "automated", 
              "efficient", "powerful", "innovative", "cutting-edge", 
              "state-of-the-art", "next-generation", "high-performance", 
              "scalable", "robust", "flexible", "adaptive"]

BENEFITS = ["increasing accuracy", "reducing errors", "improving efficiency", 
            "saving time", "cutting costs", "enhancing productivity", 
            "boosting performance", "minimizing risks", "maximizing returns", 
            "streamlining operations", "automating processes", 
            "optimizing resources", "facilitating decision-making"]

# Text generation functions
def random_sentence(min_words: int = 5, max_words: int = 15) -> str:
    """Generate a random sentence about AI/ML."""
    topic = random.choice(TOPICS)
    adj = random.choice(ADJECTIVES)
    verb = random.choice(VERBS)
    benefit = random.choice(BENEFITS)
    
    sentence = f"{adj} {topic} {verb} data by {benefit}"
    
    # Add random complexity
    if random.random() < 0.3:
        connector = random.choice(CONJUNCTIONS)
        second_topic = random.choice(TOPICS)
        second_verb = random.choice(VERBS)
        sentence += f" {connector} {second_topic} {second_verb} information"
    
    return sentence

def generate_paragraph(num_sentences: int = 3) -> str:
    """Generate a paragraph with a random number of sentences."""
    sentences = [random_sentence() for _ in range(num_sentences)]
    return " ".join(sentences)

def generate_query(topic: Optional[str] = None) -> str:
    """Generate a query about a topic."""
    if topic is None:
        topic = random.choice(TOPICS)
    
    templates = [
        f"How does {topic} work?",
        f"What is {topic}?",
        f"Benefits of {topic}",
        f"Why is {topic} important?",
        f"Applications of {topic}",
        f"{topic} use cases",
        f"{topic} implementation",
        f"{topic} examples",
        f"{topic} techniques",
        f"{topic} methods",
    ]
    
    return random.choice(templates)

def generate_document(min_sentences: int = 3, max_sentences: int = 7, seed_topic: Optional[str] = None) -> str:
    """Generate a document with some sentences about a seed topic if provided."""
    if seed_topic is None:
        return generate_paragraph(random.randint(min_sentences, max_sentences))
    
    # Start with a sentence about the seed topic
    sentences = [f"{random.choice(ADJECTIVES)} {seed_topic} {random.choice(VERBS)} data by {random.choice(BENEFITS)}"]
    
    # Add more sentences
    num_additional = random.randint(min_sentences - 1, max_sentences - 1)
    sentences.extend([random_sentence() for _ in range(num_additional)])
    
    # Shuffle sentences with the seed topic sentence staying at the beginning
    first_sentence = sentences[0]
    random.shuffle(sentences[1:])
    sentences = [first_sentence] + sentences[1:]
    
    # Add a connector to make it more coherent
    if len(sentences) > 1:
        sentences[1] = f"{random.choice(CONNECTORS)}, {sentences[1][0].lower() + sentences[1][1:]}"
    
    return " ".join(sentences)

def create_positive_pair(query_topic: Optional[str] = None) -> Tuple[str, str]:
    """Create a relevant query-document pair."""
    # Choose a random topic if none provided
    if query_topic is None:
        query_topic = random.choice(TOPICS)
    
    # Generate query and relevant document
    query = generate_query(query_topic)
    document = generate_document(seed_topic=query_topic)
    
    return query, document

def create_negative_pair(positive_query: str) -> Tuple[str, str]:
    """Create an irrelevant document for a query."""
    # Generate a random document (not specifically about the query topic)
    document = generate_document()
    
    return positive_query, document

def generate_synthetic_pairs(
    n_positive: int = 500, 
    n_negative_per_positive: int = 1,
    output_file: Union[str, Path] = "pairs.tsv"
) -> Path:
    """
    Generate a synthetic dataset of query-document pairs.
    
    Args:
        n_positive: Number of positive pairs to generate
        n_negative_per_positive: Number of negative pairs to generate per positive pair
        output_file: Output file path (relative to RAW_DATA_DIR)
        
    Returns:
        Path to the generated TSV file
    """
    # Setup directories
    setup_data_dirs()
    
    pairs = []
    
    # Generate positive pairs
    for _ in range(n_positive):
        query, document = create_positive_pair()
        pairs.append((query, document, 1))  # Label 1 for positive
        
        # Generate negatives for this query
        for _ in range(n_negative_per_positive):
            _, negative_doc = create_negative_pair(query)
            pairs.append((query, negative_doc, 0))  # Label 0 for negative
    
    # Shuffle the pairs
    random.shuffle(pairs)
    
    # Write to TSV file
    if isinstance(output_file, str):
        output_path = RAW_DATA_DIR / output_file
    else:
        output_path = output_file
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for query, document, label in pairs:
            # Clean up any tabs or newlines in the text
            query = query.replace('\t', ' ').replace('\n', ' ')
            document = document.replace('\t', ' ').replace('\n', ' ')
            f.write(f"{query}\t{document}\t{label}\n")
    
    print(f"Generated {len(pairs)} pairs ({n_positive} positive, {n_positive * n_negative_per_positive} negative)")
    print(f"Saved to {output_path}")
    
    return output_path

def expand_synthetic_dataset(
    input_file: Union[str, Path] = "pairs.tsv",
    output_file: Union[str, Path] = "expanded_pairs.tsv",
    expansion_factor: int = 2
) -> Path:
    """
    Expand a synthetic dataset by creating more variants of existing queries and documents.
    
    Args:
        input_file: Input file path (relative to RAW_DATA_DIR)
        output_file: Output file path (relative to RAW_DATA_DIR)
        expansion_factor: How many times larger the expanded dataset should be
        
    Returns:
        Path to the expanded TSV file
    """
    # Setup directories
    setup_data_dirs()
    
    # Resolve paths
    if isinstance(input_file, str):
        input_path = RAW_DATA_DIR / input_file
    else:
        input_path = input_file
        
    if isinstance(output_file, str):
        output_path = RAW_DATA_DIR / output_file
    else:
        output_path = output_file
    
    # Read the input file
    df = pd.read_csv(input_path, sep='\t', header=None, 
                    names=['query', 'document', 'label'])
    
    # Get positive and negative pairs
    positives = df[df['label'] == 1][['query', 'document']].values.tolist()
    
    # Create a set of existing query-document pairs to avoid duplicates
    existing_pairs = set((q, d) for q, d in zip(df['query'], df['document']))
    
    # Generate new pairs
    new_pairs = []
    while len(new_pairs) < (len(df) * expansion_factor) - len(df):
        if random.random() < 0.7:  # 70% chance of creating a variant of an existing positive
            # Select a random positive pair
            query, document = random.choice(positives)
            
            # Create a variant of the document
            topic = next((t for t in TOPICS if t in query.lower()), random.choice(TOPICS))
            new_document = generate_document(seed_topic=topic)
            
            # Add as a positive pair
            if (query, new_document) not in existing_pairs:
                new_pairs.append((query, new_document, 1))
                existing_pairs.add((query, new_document))
        else:  # 30% chance of creating a completely new pair
            if random.random() < 0.5:  # 50% chance of positive
                query, document = create_positive_pair()
                if (query, document) not in existing_pairs:
                    new_pairs.append((query, document, 1))
                    existing_pairs.add((query, document))
            else:  # 50% chance of negative
                query = random.choice(positives)[0]  # Take query from existing positives
                _, document = create_negative_pair(query)
                if (query, document) not in existing_pairs:
                    new_pairs.append((query, document, 0))
                    existing_pairs.add((query, document))
    
    # Combine original and new pairs
    all_pairs = list(zip(df['query'], df['document'], df['label'])) + new_pairs
    random.shuffle(all_pairs)
    
    # Write to TSV file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for query, document, label in all_pairs:
            # Clean up any tabs or newlines in the text
            query = query.replace('\t', ' ').replace('\n', ' ')
            document = document.replace('\t', ' ').replace('\n', ' ')
            f.write(f"{query}\t{document}\t{label}\n")
    
    print(f"Original dataset: {len(df)} pairs")
    print(f"Added {len(new_pairs)} new pairs")
    print(f"Expanded dataset: {len(all_pairs)} pairs")
    print(f"Saved to {output_path}")
    
    return output_path 
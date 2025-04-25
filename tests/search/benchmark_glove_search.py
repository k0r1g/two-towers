#!/usr/bin/env python
"""
Benchmark script for the GloVe search utility.
This script measures performance of the GloVe search with varying numbers of documents.
"""

import time
import random
import string
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to the path to import modules correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from inference.search import GloVeSearch
from twotower.utils import setup_logging
import logging

# Set up logging
setup_logging()
logger = logging.getLogger('tests.search.benchmark_glove')

# Create output directory if it doesn't exist
os.makedirs('reports', exist_ok=True)

def generate_random_document(word_count=50, vocab_size=1000):
    """Generate a random document with specified number of words."""
    words = []
    # Generate a list of random words from a limited vocabulary
    vocab = [
        ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
        for _ in range(vocab_size)
    ]
    
    # Add some common words that are likely in GloVe vocabulary
    common_words = [
        'the', 'of', 'and', 'a', 'to', 'in', 'is', 'you', 'that', 'it',
        'he', 'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they',
        'I', 'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had',
        'by', 'word', 'but', 'not', 'what', 'all', 'were', 'we', 'when',
        'your', 'can', 'said', 'there', 'use', 'an', 'each', 'which',
        'she', 'do', 'how', 'their', 'if', 'will', 'up', 'other', 'about',
        'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her', 'would',
        'make', 'like', 'him', 'into', 'time', 'has', 'look', 'two', 'more',
        'write', 'go', 'see', 'number', 'no', 'way', 'could', 'people',
        'my', 'than', 'first', 'water', 'been', 'call', 'who', 'oil',
        'its', 'now', 'find', 'long', 'down', 'day', 'did', 'get', 'come',
        'made', 'may', 'part'
    ]
    
    vocab.extend(common_words)
    
    for _ in range(word_count):
        words.append(random.choice(vocab))
        
    return ' '.join(words)

def benchmark_glove_search():
    """Benchmark GloVe search with varying numbers of documents."""
    logger.info("Starting GloVe search benchmark")
    
    # Number of documents to test
    doc_counts = [10, 50, 100, 500, 1000, 2000]
    
    # Statistics to track
    index_times = []
    search_times = []
    memory_usage = []
    
    # GloVe models to test
    glove_models = ['glove-twitter-25', 'glove-wiki-gigaword-50']
    
    for model_name in glove_models:
        model_index_times = []
        model_search_times = []
        
        logger.info(f"Testing with model: {model_name}")
        
        for count in doc_counts:
            logger.info(f"Generating {count} random documents")
            
            # Generate random documents
            docs = [generate_random_document() for _ in range(count)]
            
            # Initialize search engine
            search_engine = GloVeSearch(model_name=model_name)
            
            # Measure indexing time
            logger.info(f"Indexing {count} documents")
            start_time = time.time()
            search_engine.index_documents(docs)
            index_time = time.time() - start_time
            model_index_times.append(index_time)
            
            logger.info(f"Indexing {count} documents took {index_time:.2f} seconds")
            
            # Measure search time (average of 10 searches)
            logger.info(f"Performing searches")
            total_search_time = 0
            num_searches = 10
            
            for _ in range(num_searches):
                query = generate_random_document(word_count=5)
                start_time = time.time()
                search_engine.search(query, top_k=10)
                total_search_time += time.time() - start_time
                
            avg_search_time = total_search_time / num_searches
            model_search_times.append(avg_search_time)
            
            logger.info(f"Average search time for {count} documents: {avg_search_time:.4f} seconds")
        
        # Plot results for this model
        plt.figure(figsize=(12, 6))
        
        # Plot indexing time
        plt.subplot(1, 2, 1)
        plt.plot(doc_counts, model_index_times, marker='o')
        plt.title(f'GloVe Indexing Time ({model_name})')
        plt.xlabel('Number of Documents')
        plt.ylabel('Time (seconds)')
        plt.grid(True)
        
        # Plot search time
        plt.subplot(1, 2, 2)
        plt.plot(doc_counts, model_search_times, marker='o')
        plt.title(f'GloVe Average Search Time ({model_name})')
        plt.xlabel('Number of Documents')
        plt.ylabel('Time (seconds)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'reports/glove_search_benchmark_{model_name.replace("-", "_")}.png')
        
        # Save results to CSV
        with open(f'reports/glove_search_benchmark_{model_name.replace("-", "_")}.csv', 'w') as f:
            f.write('doc_count,index_time,search_time\n')
            for i, count in enumerate(doc_counts):
                f.write(f'{count},{model_index_times[i]},{model_search_times[i]}\n')
        
    logger.info("Benchmark completed. Reports saved to 'reports' directory")

if __name__ == "__main__":
    benchmark_glove_search() 
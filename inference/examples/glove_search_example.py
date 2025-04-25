#!/usr/bin/env python
"""
Example demonstrating the GloVeSearch functionality.
"""

import logging
from twotower.utils import setup_logging
from inference.search import GloVeSearch

# Set up logging
setup_logging()
logger = logging.getLogger('inference.examples.glove_search')

def main():
    # Sample documents
    docs = [
        "Machine learning is great for data analysis.",
        "Python programming makes handling data easy.",
        "Artificial intelligence and deep learning are popular today.",
        "Natural language processing allows computers to understand text.",
        "Dogs are very friendly animals."
    ]
    
    # Initialize GloVe search
    logger.info("Initializing GloVe search with glove-wiki-gigaword-50 model")
    search_engine = GloVeSearch(model_name='glove-wiki-gigaword-50')
    
    # Index the documents
    logger.info(f"Indexing {len(docs)} documents")
    search_engine.index_documents(docs)
    
    # Perform search
    query = "I like cats."
    logger.info(f"Searching for: '{query}'")
    results = search_engine.search(query, top_k=3)
    
    # Display results
    print(f"\nQuery: '{query}'")
    print("\nTop Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. '{result['document']}' (Score: {result['score']:.4f})")
    
    # Save the index
    search_engine.save_index("glove_search_index.pkl")
    logger.info("Saved index to glove_search_index.pkl")
    
    # Create a new search engine and load the index
    logger.info("Creating new search engine and loading index")
    new_search_engine = GloVeSearch(model_name='glove-wiki-gigaword-50')
    new_search_engine.load_index("glove_search_index.pkl")
    
    # Perform search using the loaded index
    query = "Deep learning techniques"
    logger.info(f"Searching loaded index for: '{query}'")
    results = new_search_engine.search(query, top_k=3)
    
    # Display results
    print(f"\nQuery: '{query}'")
    print("\nTop Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. '{result['document']}' (Score: {result['score']:.4f})")

if __name__ == "__main__":
    main() 
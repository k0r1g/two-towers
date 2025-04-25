#!/usr/bin/env python
"""
Command-line interface for document retrieval using Two-Tower models.

Usage:
    python -m inference.cli.retrieve build-index --model path/to/model.pt --documents path/to/docs.txt --output path/to/index.pkl
    python -m inference.cli.retrieve search --model path/to/model.pt --index path/to/index.pkl --query "search query"
"""

import argparse
import torch
import pickle
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

from twotower import setup_logging, load_checkpoint
from twotower.tokenisers import CharTokeniser, WordTokeniser
from inference.search import TwoTowerSearch

logger = logging.getLogger('inference.cli.retrieve')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Retrieve documents using a Two-Tower model")
    
    # Subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Parser for building an index
    index_parser = subparsers.add_parser('build-index', help='Build a document index')
    index_parser.add_argument('--model', required=True, help='Path to model checkpoint')
    index_parser.add_argument('--documents', required=True, help='Path to documents file (one per line)')
    index_parser.add_argument('--output', required=True, help='Path to save the index')
    index_parser.add_argument('--max-length', type=int, default=64, help='Maximum sequence length')
    index_parser.add_argument('--batch-size', type=int, default=32, help='Batch size for encoding')
    index_parser.add_argument('--tokenizer', choices=['char', 'word'], default='char', 
                            help='Type of tokenizer to use')
    
    # Parser for search/retrieval
    search_parser = subparsers.add_parser('search', help='Search documents for a query')
    search_parser.add_argument('--model', required=True, help='Path to model checkpoint')
    search_parser.add_argument('--index', required=True, help='Path to document index')
    search_parser.add_argument('--query', required=True, help='Query string')
    search_parser.add_argument('--top-k', type=int, default=5, help='Number of documents to retrieve')
    search_parser.add_argument('--tokenizer', choices=['char', 'word'], default='char', 
                            help='Type of tokenizer to use')
    
    # Common arguments
    parser.add_argument('--device', default=None, help='Device to use (cuda/cpu)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                      help='Logging level')
    
    return parser.parse_args()

def load_model_and_tokenizer(model_path, tokenizer_type, device):
    """Load the model and tokenizer from a checkpoint."""
    # Load checkpoint
    logger.info(f"Loading model from {model_path}")
    checkpoint = load_checkpoint(model_path, device=device)
    
    # Create tokenizer
    if tokenizer_type == 'char':
        tokenizer = CharTokeniser()
    else:
        tokenizer = WordTokeniser()
        
    # Load vocabulary from checkpoint
    tokenizer.string_to_index = checkpoint['vocab']
    tokenizer.index_to_string = {idx: char for char, idx in tokenizer.string_to_index.items()}
    
    # Get model from checkpoint
    model = checkpoint['model']
    
    return model, tokenizer

def build_index_command(args):
    """Command to build a document index."""
    # Determine device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model, args.tokenizer, device)
    
    # Create search engine
    search_engine = TwoTowerSearch(model, tokenizer, device)
    
    # Load documents
    with open(args.documents, 'r') as f:
        documents = [line.strip() for line in f]
    
    logger.info(f"Loaded {len(documents)} documents")
    
    # Build index
    search_engine.index_documents(documents)
    
    # Save index
    search_engine.save_index(args.output)
    
def search_command(args):
    """Command to search documents for a query."""
    # Determine device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model, args.tokenizer, device)
    
    # Create search engine
    search_engine = TwoTowerSearch(model, tokenizer, device)
    
    # Load index
    search_engine.load_index(args.index)
    
    # Perform search
    start_time = time.time()
    results = search_engine.search(args.query, top_k=args.top_k)
    retrieval_time = time.time() - start_time
    
    # Print results
    print(f"\nQuery: {args.query}")
    print(f"Retrieval time: {retrieval_time:.4f}s")
    print(f"\nTop {len(results)} results:")
    
    for i, result in enumerate(results):
        print(f"{i+1}. {result['score']:.4f} | {result['document'][:100]}...")

def main():
    """Main function for CLI usage."""
    args = parse_args()
    
    # Set up logging
    setup_logging(log_level=args.log_level)
    
    # Dispatch to appropriate command
    if args.mode == 'build-index':
        build_index_command(args)
    elif args.mode == 'search':
        search_command(args)
    else:
        print("Please specify a mode: build-index or search")
        print("Run with --help for more information")

if __name__ == "__main__":
    main() 
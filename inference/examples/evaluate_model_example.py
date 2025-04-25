#!/usr/bin/env python
"""
Example demonstrating how to evaluate a two-tower model.
"""

import logging
import torch
import argparse
from twotower.utils import setup_logging, load_checkpoint
from twotower.tokenisers import CharTokeniser, WordTokeniser
from twotower.evaluate import evaluate_model, print_evaluation_results

# Set up logging
setup_logging()
logger = logging.getLogger('inference.examples.evaluate_model')

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate a two-tower model')
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--test-data', required=True, help='Path to test data file')
    parser.add_argument('--tokenizer', choices=['char', 'word'], default='char', 
                       help='Type of tokenizer to use')
    parser.add_argument('--device', default=None, help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    # Determine device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    checkpoint = load_checkpoint(args.model, device=device)
    model = checkpoint['model']
    
    # Create tokenizer
    if args.tokenizer == 'char':
        tokenizer = CharTokeniser()
    else:
        tokenizer = WordTokeniser()
        
    # Load vocabulary from checkpoint
    tokenizer.string_to_index = checkpoint['vocab']
    tokenizer.index_to_string = {idx: char for char, idx in tokenizer.string_to_index.items()}
    
    # Load test data
    # This would normally be loaded from a file in a real application
    # For this example, we'll create some dummy test data
    logger.info("Creating sample test data")
    test_data = [
        (
            "machine learning algorithm", 
            [
                "Machine learning algorithms are used in AI applications.",
                "Deep learning is a subset of machine learning.",
                "Python is a programming language used for data science.",
                "Natural language processing helps computers understand text.",
                "Algorithms are step-by-step procedures for calculations."
            ],
            [1, 1, 0, 0, 1]  # Relevance scores (1 = relevant, 0 = irrelevant)
        ),
        (
            "python programming", 
            [
                "Python is a high-level programming language.",
                "Machine learning models can be built with Python.",
                "Java is another popular programming language.",
                "Data scientists often use Python for analysis.",
                "Algorithms are implemented in various languages."
            ],
            [1, 1, 0, 1, 0]  # Relevance scores
        )
    ]
    
    # Evaluate model
    logger.info("Evaluating model")
    results = evaluate_model(
        model=model,
        test_data=test_data,
        tokenizer=tokenizer,
        metrics=['precision', 'recall', 'mrr', 'ndcg'],
        k_values=[1, 3, 5],
        batch_size=32,
        device=device
    )
    
    # Print results
    print_evaluation_results(results)

if __name__ == "__main__":
    main() 
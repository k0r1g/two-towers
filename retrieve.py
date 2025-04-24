#!/usr/bin/env python
"""
Script for retrieving documents using a trained two-tower model.

Usage:
    python retrieve.py --model checkpoints/best_model.pt --index my_document_index.pkl --query "search query"
"""

import argparse
import torch
import pickle
import logging
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

from twotower import (
    TwoTower, CharTokeniser, load_checkpoint, setup_logging, log_tensor_info
)

logger = logging.getLogger('retrieve')

def encode_query(
    query: str,
    model: TwoTower,
    tokeniser: CharTokeniser,
    device: str,
    max_length: int = 64
) -> torch.Tensor:
    """
    Encode a query string into a vector representation.
    
    Args:
        query: Query string
        model: Two-tower model
        tokeniser: Tokeniser
        device: Device to run on
        max_length: Maximum sequence length
    
    Returns:
        Query vector
    """
    # Tokenize and pad the query
    query_encoded = tokeniser.truncate_and_pad(
        tokeniser.encode(query),
        max_length
    )
    
    # Convert to tensor and move to device
    query_tensor = torch.tensor([query_encoded], device=device)
    
    # Get query embedding
    model.eval()
    with torch.no_grad():
        query_vector = model.query_tower(query_tensor)
    
    return query_vector

def retrieve(
    query_vector: torch.Tensor,
    document_vectors: torch.Tensor,
    document_texts: List[str],
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Retrieve the top-k documents most similar to the query.
    
    Args:
        query_vector: Query vector
        document_vectors: Document vectors
        document_texts: Original document texts
        top_k: Number of documents to retrieve
    
    Returns:
        List of (document_text, similarity_score) tuples
    """
    # Compute similarities
    similarity_scores = torch.nn.functional.cosine_similarity(
        query_vector, document_vectors
    ).squeeze(0)
    
    # Get top-k indices
    top_indices = similarity_scores.argsort(descending=True)[:top_k]
    
    # Return results
    results = [
        (document_texts[idx], similarity_scores[idx].item())
        for idx in top_indices
    ]
    
    return results

def build_index(
    documents: List[str],
    model: TwoTower,
    tokeniser: CharTokeniser,
    device: str,
    max_length: int = 64,
    batch_size: int = 32
) -> torch.Tensor:
    """
    Build an index of document vectors for retrieval.
    
    Args:
        documents: List of document texts
        model: Two-tower model
        tokeniser: Tokeniser
        device: Device to use
        max_length: Maximum sequence length
        batch_size: Batch size for encoding
    
    Returns:
        Tensor of document vectors
    """
    logger.info(f"Building index for {len(documents)} documents")
    
    # Initialize empty tensor for document vectors
    model.eval()
    
    # Process in batches to avoid OOM
    document_vectors = []
    
    for i in range(0, len(documents), batch_size):
        # Get batch
        batch_docs = documents[i:i+batch_size]
        
        # Tokenize and pad
        batch_encoded = [
            tokeniser.truncate_and_pad(tokeniser.encode(doc), max_length)
            for doc in batch_docs
        ]
        
        # Convert to tensor and move to device
        batch_tensor = torch.tensor(batch_encoded, device=device)
        
        # Get document embeddings
        with torch.no_grad():
            batch_vectors = model.document_tower(batch_tensor)
        
        # Add to list
        document_vectors.append(batch_vectors)
        
        if (i // batch_size) % 10 == 0:
            logger.info(f"Processed {i+len(batch_docs)}/{len(documents)} documents")
    
    # Concatenate all document vectors
    document_vectors = torch.cat(document_vectors, dim=0)
    
    logger.info(f"Built index with shape: {document_vectors.shape}")
    return document_vectors

def save_index(
    document_vectors: torch.Tensor,
    document_texts: List[str],
    output_path: str
):
    """
    Save the document index to disk.
    
    Args:
        document_vectors: Document vectors
        document_texts: Document texts
        output_path: Path to save the index
    """
    index = {
        'vectors': document_vectors.cpu(),
        'texts': document_texts
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(index, f)
    
    logger.info(f"Saved index to {output_path}")

def load_index(index_path: str, device: str = 'cpu') -> Dict[str, Any]:
    """
    Load a document index from disk.
    
    Args:
        index_path: Path to the index file
        device: Device to load vectors to
    
    Returns:
        Dictionary with 'vectors' and 'texts' keys
    """
    with open(index_path, 'rb') as f:
        index = pickle.load(f)
    
    # Move vectors to device
    index['vectors'] = index['vectors'].to(device)
    
    logger.info(f"Loaded index from {index_path} with {len(index['texts'])} documents")
    
    return index

def main():
    """Main function for CLI usage."""
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
    
    # Parser for retrieval
    retrieve_parser = subparsers.add_parser('retrieve', help='Retrieve documents for a query')
    retrieve_parser.add_argument('--model', required=True, help='Path to model checkpoint')
    retrieve_parser.add_argument('--index', required=True, help='Path to document index')
    retrieve_parser.add_argument('--query', required=True, help='Query string')
    retrieve_parser.add_argument('--top-k', type=int, default=5, help='Number of documents to retrieve')
    
    # Common arguments
    parser.add_argument('--device', default=None, help='Device to use (cuda/cpu)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                      help='Logging level')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(log_level=args.log_level)
    
    # Determine device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    checkpoint = load_checkpoint(args.model, device=device)
    
    # Create tokeniser
    tokeniser = CharTokeniser()
    tokeniser.string_to_index = checkpoint['vocab']
    tokeniser.index_to_string = {idx: char for char, idx in tokeniser.string_to_index.items()}
    
    # Load model (create empty model structure and load weights)
    from twotower.embeddings import LookupEmbedding
    from twotower.encoders import build_two_tower
    
    # Approximate model configuration from checkpoint (this could be improved by storing config in checkpoint)
    embedding_dim = 64  # Default or infer from checkpoint
    hidden_dim = 128    # Default or infer from checkpoint
    
    # Create embedding layer
    embedding = LookupEmbedding(
        vocab_size=tokeniser.vocab_size,
        embedding_dim=embedding_dim
    )
    
    # Create model
    model = build_two_tower(
        tower_name="mean",
        embedding=embedding,
        hidden_dim=hidden_dim
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Perform operation based on mode
    if args.mode == 'build-index':
        # Load documents
        with open(args.documents, 'r') as f:
            documents = [line.strip() for line in f]
        
        logger.info(f"Loaded {len(documents)} documents")
        
        # Build index
        document_vectors = build_index(
            documents=documents,
            model=model,
            tokeniser=tokeniser,
            device=device,
            max_length=args.max_length,
            batch_size=args.batch_size
        )
        
        # Save index
        save_index(
            document_vectors=document_vectors,
            document_texts=documents,
            output_path=args.output
        )
        
    elif args.mode == 'retrieve':
        # Load index
        index = load_index(args.index, device=device)
        
        # Encode query
        start_time = time.time()
        query_vector = encode_query(
            query=args.query,
            model=model,
            tokeniser=tokeniser,
            device=device
        )
        
        # Retrieve documents
        results = retrieve(
            query_vector=query_vector,
            document_vectors=index['vectors'],
            document_texts=index['texts'],
            top_k=args.top_k
        )
        
        # Calculate retrieval time
        retrieval_time = time.time() - start_time
        
        # Print results
        print(f"\nQuery: {args.query}")
        print(f"Retrieval time: {retrieval_time:.4f}s")
        print(f"\nTop {len(results)} results:")
        
        for i, (doc_text, score) in enumerate(results):
            print(f"{i+1}. {score:.4f} | {doc_text[:100]}...")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 
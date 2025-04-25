"""
Model evaluation metrics and utilities for Two-Tower models.
"""

import numpy as np
from typing import List, Dict, Any, Union, Tuple, Callable
import torch
import logging
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    average_precision_score, ndcg_score
)

logger = logging.getLogger('twotower.evaluate')

def mean_reciprocal_rank(
    relevance_scores: Union[List[float], np.ndarray]
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    Args:
        relevance_scores: Binary relevance scores (1 = relevant, 0 = irrelevant)
        
    Returns:
        MRR score
    """
    relevance_scores = np.asarray(relevance_scores)
    
    # Find the first relevant document
    relevant_indices = np.where(relevance_scores == 1)[0]
    if len(relevant_indices) == 0:
        return 0.0
    
    # Calculate reciprocal rank (1-indexed)
    first_relevant_rank = relevant_indices[0] + 1
    return 1.0 / first_relevant_rank

def precision_at_k(
    relevance_scores: Union[List[float], np.ndarray],
    k: int
) -> float:
    """
    Calculate Precision@K.
    
    Args:
        relevance_scores: Binary relevance scores (1 = relevant, 0 = irrelevant)
        k: Number of top results to consider
        
    Returns:
        Precision@K score
    """
    relevance_scores = np.asarray(relevance_scores)
    
    # Take top-k scores
    if len(relevance_scores) < k:
        # Pad with zeros if not enough results
        top_k_scores = np.pad(relevance_scores, (0, k - len(relevance_scores)))
    else:
        top_k_scores = relevance_scores[:k]
    
    # Calculate precision
    return np.mean(top_k_scores)

def recall_at_k(
    relevance_scores: Union[List[float], np.ndarray],
    k: int,
    total_relevant: int
) -> float:
    """
    Calculate Recall@K.
    
    Args:
        relevance_scores: Binary relevance scores (1 = relevant, 0 = irrelevant)
        k: Number of top results to consider
        total_relevant: Total number of relevant documents
        
    Returns:
        Recall@K score
    """
    if total_relevant == 0:
        return 0.0
        
    relevance_scores = np.asarray(relevance_scores)
    
    # Take top-k scores
    if len(relevance_scores) < k:
        top_k_scores = relevance_scores
    else:
        top_k_scores = relevance_scores[:k]
    
    # Calculate recall
    return np.sum(top_k_scores) / total_relevant

def ndcg_at_k(
    relevance_scores: Union[List[float], np.ndarray],
    k: int
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K (NDCG@K).
    
    Args:
        relevance_scores: Relevance scores (higher = more relevant)
        k: Number of top results to consider
        
    Returns:
        NDCG@K score
    """
    relevance_scores = np.asarray(relevance_scores)
    
    # Use scikit-learn's implementation
    y_true = np.sort(relevance_scores)[::-1]  # Ideal ranking
    y_score = relevance_scores
    
    # Handle case with not enough results
    if len(y_true) < k:
        y_true = np.pad(y_true, (0, k - len(y_true)))
        y_score = np.pad(y_score, (0, k - len(y_score)))
    
    # Reshape for sklearn's expected format
    y_true = y_true.reshape(1, -1)
    y_score = y_score.reshape(1, -1)
    
    return ndcg_score(y_true, y_score, k=k)

def evaluate_model(
    model,
    test_data: List[Tuple[str, List[str], List[int]]],
    tokenizer,
    metrics: List[str] = ['precision', 'recall', 'mrr', 'ndcg'],
    k_values: List[int] = [1, 5, 10],
    batch_size: int = 32,
    device: str = 'cpu',
) -> Dict[str, float]:
    """
    Evaluate a model on test data.
    
    Args:
        model: The two-tower model to evaluate
        test_data: List of (query, documents, relevance_scores) tuples
        tokenizer: Tokenizer for preprocessing text
        metrics: List of metrics to compute
        k_values: List of k values for precision@k, recall@k, etc.
        batch_size: Batch size for processing
        device: Device to run evaluation on
        
    Returns:
        Dictionary of metric scores
    """
    model.eval()
    model = model.to(device)
    results = {}
    
    all_precision = []
    all_recall = []
    all_mrr = []
    all_ndcg = []
    
    for query, documents, relevance in test_data:
        # Encode query
        query_encoded = tokenizer.encode(query)
        query_padded = tokenizer.truncate_and_pad(query_encoded, 64)
        query_tensor = torch.tensor([query_padded], device=device)
        
        # Get query embedding
        with torch.no_grad():
            query_embedding = model.query_tower(query_tensor)
        
        # Process documents in batches
        doc_embeddings = []
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            
            # Encode documents
            docs_encoded = [tokenizer.encode(doc) for doc in batch_docs]
            docs_padded = [tokenizer.truncate_and_pad(enc, 64) for enc in docs_encoded]
            docs_tensor = torch.tensor(docs_padded, device=device)
            
            # Get document embeddings
            with torch.no_grad():
                batch_embeddings = model.document_tower(docs_tensor)
                doc_embeddings.append(batch_embeddings)
        
        # Combine document embeddings
        doc_embeddings = torch.cat(doc_embeddings, dim=0)
        
        # Calculate similarity scores
        similarity_scores = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(1),
            doc_embeddings.unsqueeze(0),
            dim=2
        ).squeeze(0)
        
        # Sort documents by similarity
        _, sorted_indices = torch.sort(similarity_scores, descending=True)
        sorted_indices = sorted_indices.cpu().numpy()
        
        # Reorder relevance scores based on similarity ranking
        sorted_relevance = np.array(relevance)[sorted_indices]
        
        # Calculate metrics
        total_relevant = np.sum(relevance)
        
        # Precision at different k values
        precision_values = [precision_at_k(sorted_relevance, k) for k in k_values]
        all_precision.append(precision_values)
        
        # Recall at different k values
        recall_values = [recall_at_k(sorted_relevance, k, total_relevant) for k in k_values]
        all_recall.append(recall_values)
        
        # Mean Reciprocal Rank
        mrr = mean_reciprocal_rank(sorted_relevance)
        all_mrr.append(mrr)
        
        # NDCG at different k values
        ndcg_values = [ndcg_at_k(sorted_relevance, k) for k in k_values]
        all_ndcg.append(ndcg_values)
    
    # Aggregate metrics across all queries
    if 'precision' in metrics:
        for i, k in enumerate(k_values):
            results[f'precision@{k}'] = np.mean([p[i] for p in all_precision])
    
    if 'recall' in metrics:
        for i, k in enumerate(k_values):
            results[f'recall@{k}'] = np.mean([r[i] for r in all_recall])
    
    if 'mrr' in metrics:
        results['mrr'] = np.mean(all_mrr)
    
    if 'ndcg' in metrics:
        for i, k in enumerate(k_values):
            results[f'ndcg@{k}'] = np.mean([n[i] for n in all_ndcg])
    
    return results

def print_evaluation_results(results: Dict[str, float]) -> None:
    """
    Print evaluation results in a formatted way.
    
    Args:
        results: Dictionary of metric scores
    """
    print("\nEvaluation Results:")
    print("="*50)
    
    precision_metrics = {k: v for k, v in results.items() if k.startswith('precision')}
    if precision_metrics:
        print("\nPrecision:")
        for k, v in sorted(precision_metrics.items()):
            print(f"  {k}: {v:.4f}")
    
    recall_metrics = {k: v for k, v in results.items() if k.startswith('recall')}
    if recall_metrics:
        print("\nRecall:")
        for k, v in sorted(recall_metrics.items()):
            print(f"  {k}: {v:.4f}")
    
    if 'mrr' in results:
        print("\nMean Reciprocal Rank:")
        print(f"  MRR: {results['mrr']:.4f}")
    
    ndcg_metrics = {k: v for k, v in results.items() if k.startswith('ndcg')}
    if ndcg_metrics:
        print("\nNDCG:")
        for k, v in sorted(ndcg_metrics.items()):
            print(f"  {k}: {v:.4f}")
    
    print("="*50) 
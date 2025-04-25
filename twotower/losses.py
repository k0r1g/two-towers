import torch
import torch.nn.functional as F
import logging
from typing import Dict, Callable

# Set up logging
logger = logging.getLogger('twotower.losses')

def contrastive_triplet_loss(
    q_emb: torch.Tensor,
    d_pos_emb: torch.Tensor,
    d_neg_emb: torch.Tensor,
    margin: float = 0.2
) -> torch.Tensor:
    """
    Contrastive triplet margin loss for training two-tower models.
    
    Args:
        q_emb: Query embeddings of shape (batch_size, hidden_dim)
        d_pos_emb: Positive document embeddings of shape (batch_size, hidden_dim)
        d_neg_emb: Negative document embeddings of shape (batch_size, hidden_dim)
        margin: Margin parameter for the hinge loss (default: 0.2)
    
    Returns:
        Loss tensor (scalar)
    """
    # Calculate cosine similarities
    sim_pos = F.cosine_similarity(q_emb, d_pos_emb, dim=1)  # (batch_size,)
    sim_neg = F.cosine_similarity(q_emb, d_neg_emb, dim=1)  # (batch_size,)
    
    # Calculate per-sample loss using hinge-margin: max(0, margin - sim_pos + sim_neg)
    per_sample_loss = F.relu(margin - sim_pos + sim_neg)
    
    # Calculate mean loss over the batch
    loss = per_sample_loss.mean()
    
    # Log similarity statistics for debugging
    if logger.isEnabledFor(logging.DEBUG):
        pos_avg = sim_pos.mean().item()
        neg_avg = sim_neg.mean().item()
        diff_avg = (sim_pos - sim_neg).mean().item()
        logger.debug(f"Similarities - Pos: {pos_avg:.4f}, Neg: {neg_avg:.4f}, Diff: {diff_avg:.4f}")
    
    return loss


def multiple_negatives_loss(
    q_emb: torch.Tensor,
    d_pos_emb: torch.Tensor,
    d_neg_embs: torch.Tensor,
    temperature: float = 0.1
) -> torch.Tensor:
    """
    InfoNCE-style loss with multiple negatives for each query.
    
    Args:
        q_emb: Query embeddings of shape (batch_size, hidden_dim)
        d_pos_emb: Positive document embeddings of shape (batch_size, hidden_dim)
        d_neg_embs: Negative document embeddings of shape (batch_size, num_negatives, hidden_dim)
        temperature: Temperature parameter for scaling logits (default: 0.1)
    
    Returns:
        Loss tensor (scalar)
    """
    batch_size, num_negatives, hidden_dim = d_neg_embs.shape
    
    # Reshape for computation
    q_emb_expanded = q_emb.unsqueeze(1).expand(-1, num_negatives + 1, -1)  # (B, N+1, H)
    
    # Combine positive and negative documents
    d_embs = torch.cat([d_pos_emb.unsqueeze(1), d_neg_embs], dim=1)  # (B, N+1, H)
    
    # Compute similarities
    similarities = F.cosine_similarity(q_emb_expanded, d_embs, dim=2)  # (B, N+1)
    
    # Scale by temperature
    logits = similarities / temperature
    
    # The positive example is at index 0
    labels = torch.zeros(batch_size, dtype=torch.long, device=q_emb.device)
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(logits, labels)
    
    return loss


def in_batch_sampled_softmax_loss(
    q_emb: torch.Tensor,
    d_emb: torch.Tensor,
    temperature: float = 0.1
) -> torch.Tensor:
    """
    In-batch sampled softmax loss, using all other documents in batch as negatives.
    
    Args:
        q_emb: Query embeddings of shape (batch_size, hidden_dim)
        d_emb: Document embeddings of shape (batch_size, hidden_dim)
        temperature: Temperature parameter for scaling logits (default: 0.1)
    
    Returns:
        Loss tensor (scalar)
    """
    batch_size = q_emb.shape[0]
    
    # Calculate similarities between all queries and all documents
    similarities = torch.matmul(q_emb, d_emb.transpose(0, 1))  # (B, B)
    
    # Scale by temperature
    logits = similarities / temperature
    
    # The positive examples are on the diagonal
    labels = torch.arange(batch_size, device=q_emb.device)
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(logits, labels)
    
    return loss


# Registry of available loss functions
LOSS_REGISTRY = {
    "triplet": contrastive_triplet_loss,
    "multiple_negatives": multiple_negatives_loss,
    "in_batch": in_batch_sampled_softmax_loss,
    # Add more loss functions here
}

def build(name: str, **kwargs) -> Callable:
    """
    Build a loss function by name from the registry
    
    Args:
        name: Name of the loss function to build
        **kwargs: Additional arguments to pass to the loss function
    
    Returns:
        A callable loss function with default parameters set
    """
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss function: {name}. Available options: {list(LOSS_REGISTRY.keys())}")
    
    loss_fn = LOSS_REGISTRY[name]
    
    # Return a partial function with the specified parameters
    if kwargs:
        from functools import partial
        return partial(loss_fn, **kwargs)
    
    return loss_fn 
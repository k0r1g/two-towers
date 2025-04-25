import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Tuple, Optional

from .embeddings import BaseEmbedding

# Set up logging
logger = logging.getLogger('twotower.encoders')

class BaseTower(nn.Module):
    """Base class for tower/encoder architectures"""
    def __init__(self, embedding: BaseEmbedding, hidden_dim: int):
        super().__init__()
        self.embedding = embedding
        self.hidden_dim = hidden_dim
    
    def log_params(self):
        """Log parameter information for this tower"""
        tower_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Tower parameters: {tower_params:,}")


class MeanPoolingTower(BaseTower):
    """
    Tower using mean pooling over token embeddings with feedforward network.
    This is the original Tower implementation from two_tower_mini.py.
    """
    def __init__(self, embedding: BaseEmbedding, hidden_dim: int):
        super().__init__(embedding, hidden_dim)
        logger.info(f"Initializing MeanPoolingTower with hidden_dim={hidden_dim}")
        
        # Embedding dim is available through the embedding layer
        embedding_dim = embedding.embedding_dim
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Count parameters
        ff_params = embedding_dim * hidden_dim + hidden_dim + hidden_dim * hidden_dim + hidden_dim
        logger.info(f"Tower feed-forward parameters: {ff_params:,}")
        
        self.log_params()
    
    def forward(self, input_ids):
        """
        Args:
            input_ids: Token IDs tensor of shape (batch_size, sequence_length)
        
        Returns:
            Encoded representation of shape (batch_size, hidden_dim)
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Tower input_ids shape: {input_ids.shape}")
        
        # Create mask for padding tokens
        mask = (input_ids > 0).float().unsqueeze(-1)  # (batch_size, sequence_length, 1)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Token mask shape: {mask.shape}")
        
        # Get embeddings and apply mask
        embeddings = self.embedding(input_ids) * mask  # (batch_size, sequence_length, embedding_dim)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Embeddings shape: {embeddings.shape}")
        
        # Mean pooling (accounting for variable sequence lengths)
        pooled = embeddings.sum(1) / (mask.sum(1) + 1e-9)  # (batch_size, embedding_dim)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Pooled embeddings shape: {pooled.shape}")
        
        # Apply feed-forward and normalize
        output = F.normalize(self.feed_forward(pooled), dim=-1)  # (batch_size, hidden_dim)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Tower output shape: {output.shape}")
        
        return output


class AveragePoolingTower(BaseTower):
    """
    Simple average pooling encoder that converts token embeddings to a fixed-size representation.
    Optional projection layer for dimensionality reduction.
    """
    def __init__(self, embedding: BaseEmbedding, hidden_dim: int, dropout: float = 0.1):
        super().__init__(embedding, hidden_dim)
        logger.info(f"Initializing AveragePoolingTower with hidden_dim={hidden_dim}, dropout={dropout}")
        
        # Embedding dim is available through the embedding layer
        embedding_dim = embedding.embedding_dim
        
        # Optional projection layer if output_dim != embedding_dim
        self.has_projection = (self.hidden_dim != embedding_dim)
        if self.has_projection:
            logger.info(f"Adding projection layer from {embedding_dim} to {hidden_dim}")
            self.projection = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            )
        
        # Count parameters
        if self.has_projection:
            proj_params = embedding_dim * hidden_dim + hidden_dim * 2  # weight matrix + layernorm params
            logger.info(f"Projection layer parameters: {proj_params:,}")
        
        self.log_params()
    
    def forward(self, input_ids):
        """
        Args:
            input_ids: Token IDs tensor of shape (batch_size, sequence_length)
        
        Returns:
            Encoded representation of shape (batch_size, hidden_dim)
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Tower input_ids shape: {input_ids.shape}")
        
        # Create mask for padding tokens
        attention_mask = (input_ids > 0).float()  # (batch_size, sequence_length)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Attention mask shape: {attention_mask.shape}")
        
        # Get embeddings
        embeddings = self.embedding(input_ids)  # (batch_size, sequence_length, embedding_dim)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Embeddings shape: {embeddings.shape}")
        
        # Compute mean only over valid tokens
        mask_expanded = attention_mask.unsqueeze(-1)  # (batch_size, sequence_length, 1)
        sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)  # (batch_size, embedding_dim)
        sum_mask = torch.sum(mask_expanded, dim=1)  # (batch_size, 1)
        pooled = sum_embeddings / (sum_mask + 1e-9)  # Avoid division by zero
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Pooled embeddings shape: {pooled.shape}")
        
        # Apply projection if needed
        if self.has_projection:
            output = self.projection(pooled)
        else:
            output = pooled
        
        # Normalize output
        output = F.normalize(output, dim=-1)
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Tower output shape: {output.shape}")
        
        return output


class TwoTower(nn.Module):
    """
    Two-Tower model consisting of separate query and document encoder towers.
    """
    def __init__(self, query_tower: BaseTower, document_tower: BaseTower = None, tied_weights: bool = False):
        """
        Args:
            query_tower: Tower for encoding queries
            document_tower: Tower for encoding documents (if None and tied_weights=True, uses query_tower)
            tied_weights: Whether to use the same tower for both queries and documents
        """
        super().__init__()
        
        self.query_tower = query_tower
        
        if tied_weights:
            logger.info("Using tied weights between query and document towers")
            self.document_tower = query_tower
        else:
            logger.info("Using separate weights for query and document towers")
            self.document_tower = document_tower if document_tower is not None else query_tower
        
        # Count total parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total trainable parameters: {total_params:,}")
    
    def forward(self, query_input, document_input=None, negative_input=None):
        """
        Forward pass for the two-tower model.
        
        Args:
            query_input: Token IDs tensor for queries of shape (batch_size, sequence_length)
            document_input: Token IDs tensor for documents of shape (batch_size, sequence_length)
            negative_input: Optional token IDs tensor for negative samples (batch_size, sequence_length)
        
        Returns:
            If negative_input is None:
                query_vector, document_vector: Encoded representations
            If negative_input is provided:
                query_vector, document_vector, negative_vector: Encoded representations
        """
        logger.debug("TwoTower forward pass")
        
        # Encode query
        query_vector = self.query_tower(query_input)
        
        # Encode positive document if provided
        document_vector = self.document_tower(document_input) if document_input is not None else None
        
        # Encode negative document if provided
        negative_vector = self.document_tower(negative_input) if negative_input is not None else None
        
        # Return based on what was provided
        if negative_vector is not None:
            return query_vector, document_vector, negative_vector
        elif document_vector is not None:
            return query_vector, document_vector
        else:
            return query_vector

    def encode_query(self, query_input):
        """Encode a query input"""
        return self.query_tower(query_input)
    
    def encode_document(self, document_input):
        """Encode a document input"""
        return self.document_tower(document_input)


# Registry of available tower architectures
TOWER_REGISTRY = {
    "mean": MeanPoolingTower,
    "avg_pool": AveragePoolingTower,
    # Add more tower architectures here
}

def build_tower(name: str, embedding: BaseEmbedding, **kwargs) -> BaseTower:
    """
    Build a tower by name from the registry
    
    Args:
        name: Name of the tower architecture to build
        embedding: Embedding layer to use
        **kwargs: Additional arguments specific to the tower architecture
    
    Returns:
        An instance of BaseTower
    """
    if name not in TOWER_REGISTRY:
        raise ValueError(f"Unknown tower architecture: {name}. Available options: {list(TOWER_REGISTRY.keys())}")
    
    return TOWER_REGISTRY[name](embedding=embedding, **kwargs)

def build_two_tower(tower_name: str, embedding: BaseEmbedding, hidden_dim: int, tied_weights: bool = False, **kwargs) -> TwoTower:
    """
    Build a complete two-tower model
    
    Args:
        tower_name: Name of the tower architecture to use
        embedding: Embedding layer to use
        hidden_dim: Hidden dimension size for the towers
        tied_weights: Whether to use the same tower for both queries and documents
        **kwargs: Additional arguments passed to the tower constructor
    
    Returns:
        An instance of TwoTower
    """
    query_tower = build_tower(tower_name, embedding, hidden_dim=hidden_dim, **kwargs)
    
    if tied_weights:
        document_tower = None
    else:
        document_tower = build_tower(tower_name, embedding, hidden_dim=hidden_dim, **kwargs)
    
    return TwoTower(query_tower, document_tower, tied_weights=tied_weights) 
from abc import ABC
import torch
import torch.nn as nn
import logging
from typing import Dict, Any

# Set up logging
logger = logging.getLogger('twotower.embeddings')

class BaseEmbedding(nn.Module, ABC):
    """Base class for all embedding layers"""
    def __init__(self, vocab_size: int, embedding_dim: int, padding_idx: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
    def log_params(self):
        """Log parameter information for this embedding layer"""
        embedding_params = self.vocab_size * self.embedding_dim
        logger.info(f"Embedding parameters: {embedding_params:,}")


class LookupEmbedding(BaseEmbedding):
    """Simple lookup table embedding (standard nn.Embedding)"""
    def __init__(self, vocab_size: int, embedding_dim: int, padding_idx: int = 0):
        super().__init__(vocab_size, embedding_dim, padding_idx)
        logger.info(f"Initializing LookupEmbedding with vocab_size={vocab_size}, embedding_dim={embedding_dim}")
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.log_params()
        
    def forward(self, input_ids):
        """
        Args:
            input_ids: Token IDs tensor of shape (batch_size, sequence_length)
        Returns:
            Embedded representation of shape (batch_size, sequence_length, embedding_dim)
        """
        return self.embedding(input_ids)


class FrozenWord2Vec(BaseEmbedding):
    """
    Loads pre-trained word vectors from Gensim KeyedVectors and freezes them.
    """
    def __init__(self, kv_path: str, vocab_size: int = None, embedding_dim: int = None, padding_idx: int = 0):
        try:
            import gensim
            import numpy as np
        except ImportError:
            raise ImportError("Please install gensim to use FrozenWord2Vec embedding: pip install gensim")
        
        logger.info(f"Loading word vectors from {kv_path}")
        self.kv = gensim.models.KeyedVectors.load(kv_path, mmap='r')
        
        # If vocab_size or embedding_dim are not specified, infer from the loaded vectors
        vocab_size = len(self.kv) + 1 if vocab_size is None else vocab_size
        embedding_dim = self.kv.vector_size if embedding_dim is None else embedding_dim
        
        super().__init__(vocab_size, embedding_dim, padding_idx)
        
        # Create tensor from word vectors with padding token at index 0
        pad_vector = torch.zeros(1, self.kv.vector_size)
        word_vectors = torch.tensor(np.array(self.kv.vectors), dtype=torch.float)
        all_vectors = torch.cat([pad_vector, word_vectors], dim=0)
        
        self.embedding = nn.Embedding.from_pretrained(
            all_vectors, 
            freeze=True, 
            padding_idx=padding_idx
        )
        
        logger.info(f"Loaded {len(self.kv)} word vectors with dimension {self.kv.vector_size}")
        self.log_params()
    
    def forward(self, input_ids):
        """
        Args:
            input_ids: Token IDs tensor of shape (batch_size, sequence_length)
        Returns:
            Embedded representation of shape (batch_size, sequence_length, embedding_dim)
        """
        return self.embedding(input_ids)


# Registry of available embeddings
REGISTRY = {
    "lookup": LookupEmbedding,
    "word2vec": FrozenWord2Vec,
    # Add more embedding types here
}

def build(name: str, vocab_size: int, **kwargs) -> BaseEmbedding:
    """
    Build an embedding layer by name from the registry
    
    Args:
        name: Name of the embedding type to build
        vocab_size: Size of the vocabulary
        **kwargs: Additional arguments specific to the embedding type
    
    Returns:
        An instance of BaseEmbedding
    """
    if name not in REGISTRY:
        raise ValueError(f"Unknown embedding: {name}. Available options: {list(REGISTRY.keys())}")
    
    return REGISTRY[name](vocab_size=vocab_size, **kwargs) 
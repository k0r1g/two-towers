# GloVe Integration Plan for Two-Tower Architecture

## Overview

This plan outlines how to integrate GloVe word embeddings from gensim into the existing two-tower architecture, supporting semantic similarity search functionality.

## Implementation Plan

### 1. Add GloVe Embedding Class (twotower/embeddings.py)

Add a new embedding class `GloVeEmbedding` that inherits from `BaseEmbedding`:

```python
class GloVeEmbedding(BaseEmbedding):
    """
    Loads pre-trained GloVe word vectors from gensim-data and provides embedding functionality.
    Can be set to frozen (non-trainable) or fine-tunable.
    """
    def __init__(
        self, 
        vocab_size: int, 
        embedding_dim: int = None, 
        model_name: str = 'glove-wiki-gigaword-50', 
        trainable: bool = False, 
        padding_idx: int = 0
    ):
        try:
            import gensim.downloader as api
            import numpy as np
        except ImportError:
            raise ImportError("Please install gensim to use GloVeEmbedding: pip install gensim")
        
        logger.info(f"Loading GloVe word vectors: {model_name}")
        self.model = api.load(model_name)
        
        # Set embedding dimension based on loaded model if not specified
        if embedding_dim is None:
            embedding_dim = self.model.vector_size
        elif embedding_dim != self.model.vector_size:
            logger.warning(f"Specified embedding_dim ({embedding_dim}) doesn't match GloVe model dimension ({self.model.vector_size})")
            
        super().__init__(vocab_size, embedding_dim, padding_idx)
        
        # Create embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        
        # Initialize embedding weights for known words
        self.init_embeddings()
        
        # Set embeddings as trainable or frozen
        self.embedding.weight.requires_grad = trainable
        
        logger.info(f"Loaded GloVe embeddings with dimension {embedding_dim} (trainable: {trainable})")
        self.log_params()
        
    def init_embeddings(self):
        """Initialize embedding weights with GloVe vectors for known words"""
        # This would be implemented based on your tokenizer vocabulary mapping
        pass
    
    def forward(self, input_ids):
        """
        Args:
            input_ids: Token IDs tensor of shape (batch_size, sequence_length)
        Returns:
            Embedded representation of shape (batch_size, sequence_length, embedding_dim)
        """
        return self.embedding(input_ids)
```

Update the `REGISTRY` in embeddings.py to include the new GloVe embedding:

```python
REGISTRY = {
    "lookup": LookupEmbedding,
    "word2vec": FrozenWord2Vec,
    "glove": GloVeEmbedding,
    # Add more embedding types here
}
```

### 2. Add Average Pooling Encoder (twotower/encoders.py)

Ensure there's a dedicated average pooling encoder that can process GloVe embeddings:

```python
class AveragePoolingTower(nn.Module):
    """
    Simple average pooling encoder that converts token embeddings to a fixed-size representation.
    """
    def __init__(self, embedding_dim, output_dim=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim or embedding_dim
        
        # Optional projection layer if output_dim != embedding_dim
        self.has_projection = (self.output_dim != self.embedding_dim)
        if self.has_projection:
            self.projection = nn.Linear(self.embedding_dim, self.output_dim)
    
    def forward(self, embeddings, attention_mask=None):
        """
        Args:
            embeddings: Token embeddings of shape (batch_size, seq_len, embedding_dim)
            attention_mask: Mask of shape (batch_size, seq_len) where 1 indicates valid tokens
            
        Returns:
            Pooled representation of shape (batch_size, output_dim)
        """
        if attention_mask is None:
            # If no mask provided, average all tokens
            pooled = embeddings.mean(dim=1)
        else:
            # Compute mean only over valid tokens
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
            sum_mask = torch.sum(mask_expanded, dim=1)
            pooled = sum_embeddings / (sum_mask + 1e-9)  # Avoid division by zero
            
        if self.has_projection:
            pooled = self.projection(pooled)
            
        return pooled
```

### 3. Add Direct Search Functionality (twotower/search.py)

Create a new module to handle direct semantic search functionality:

```python
"""
Semantic search functionality for the Two-Tower architecture.
"""

import torch
import numpy as np
from typing import List, Dict, Union, Optional
import logging

logger = logging.getLogger('twotower.search')

class SemanticSearch:
    """
    Semantic search using the query tower of a trained Two-Tower model.
    """
    def __init__(self, model, tokenizer):
        """
        Initialize the search engine with a trained Two-Tower model.
        
        Args:
            model: A trained TwoTower model
            tokenizer: The tokenizer used for preprocessing text
        """
        self.model = model
        self.tokenizer = tokenizer
        self.document_embeddings = None
        self.documents = None
        
    def index_documents(self, documents: List[str]):
        """
        Create and store embeddings for a list of documents.
        
        Args:
            documents: List of document strings to index
        """
        self.documents = documents
        self.model.eval()
        
        # Process documents in batches to avoid memory issues
        batch_size = 32
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                # Tokenize batch
                inputs = self.tokenizer(batch_docs)
                # Get document embeddings from the document tower
                doc_embeddings = self.model.encode_document(inputs)
                all_embeddings.append(doc_embeddings)
        
        # Concatenate all embeddings
        self.document_embeddings = torch.cat(all_embeddings, dim=0)
        logger.info(f"Indexed {len(documents)} documents")
        
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Union[str, float]]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: The search query string
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with document and similarity score
        """
        if self.document_embeddings is None:
            raise ValueError("No documents indexed. Call index_documents() first.")
        
        self.model.eval()
        
        # Tokenize query
        query_inputs = self.tokenizer([query])
        
        # Get query embedding
        with torch.no_grad():
            query_embedding = self.model.encode_query(query_inputs)
        
        # Calculate similarity scores
        similarity_scores = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(1),
            self.document_embeddings.unsqueeze(0),
            dim=2
        ).squeeze(0)
        
        # Get top-k results
        top_scores, top_indices = torch.topk(similarity_scores, min(top_k, len(self.documents)))
        
        # Create result list
        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            results.append({
                "document": self.documents[idx],
                "score": score
            })
            
        return results
```

### 4. Add Direct GloVe Search Utility (twotower/glove_search.py)

Create a standalone GloVe-based search utility that doesn't require the full two-tower model:

```python
"""
Standalone GloVe-based semantic search utility.
"""

import numpy as np
from typing import List, Dict, Any, Union
import logging
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger('twotower.glove_search')

class GloVeSearch:
    """
    Simple semantic search using GloVe embeddings with average pooling.
    """
    def __init__(self, model_name='glove-wiki-gigaword-50'):
        """
        Initialize the GloVe search engine.
        
        Args:
            model_name: Name of the gensim GloVe model to load
        """
        try:
            import gensim.downloader as api
        except ImportError:
            raise ImportError("Please install gensim to use GloVeSearch: pip install gensim")
            
        logger.info(f"Loading GloVe model: {model_name}")
        self.model = api.load(model_name)
        self.vector_size = self.model.vector_size
        self.document_embeddings = None
        self.documents = None
    
    def _average_pool(self, text: str) -> np.ndarray:
        """
        Create document embedding by averaging word vectors.
        
        Args:
            text: Input text string
            
        Returns:
            Numpy array of the averaged embedding
        """
        words = text.lower().split()
        word_vectors = [self.model[w] for w in words if w in self.model]
        
        if not word_vectors:
            return np.zeros(self.vector_size)
            
        return np.mean(word_vectors, axis=0)
    
    def index_documents(self, documents: List[str]):
        """
        Create and store embeddings for a list of documents.
        
        Args:
            documents: List of document strings to index
        """
        self.documents = documents
        self.document_embeddings = np.array([self._average_pool(doc) for doc in documents])
        logger.info(f"Indexed {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Union[str, float]]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: The search query string
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with document and similarity score
        """
        if self.document_embeddings is None:
            raise ValueError("No documents indexed. Call index_documents() first.")
            
        # Create query embedding
        query_embedding = self._average_pool(query).reshape(1, -1)
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity(query_embedding, self.document_embeddings)[0]
        
        # Get top-k results
        top_indices = similarity_scores.argsort()[-top_k:][::-1]
        
        # Create result list
        results = []
        for idx in top_indices:
            results.append({
                "document": self.documents[idx],
                "score": similarity_scores[idx]
            })
            
        return results
```

### 5. Add Configuration (configs/glove_search.yml)

Create a dedicated configuration file for GloVe embeddings:

```yaml
# GloVe Semantic Search Configuration
# 
# Configuration that uses pretrained GloVe embeddings with mean pooling.
# Optimized for semantic search applications.

extends: default_config.yml

#==============================================================================
# Tokenizer Configuration
#==============================================================================
tokeniser:
  type: word                                  # Word-level tokenization
  max_len: 32                                 # Maximum number of words per sequence
  lowercase: true                             # Convert text to lowercase
  strip_punctuation: true                     # Remove punctuation during tokenization
  
#==============================================================================
# Model Architecture
#==============================================================================
# Embedding configuration  
embedding:
  type: glove                                 # Use GloVe embeddings
  model_name: glove-wiki-gigaword-50          # GloVe model to use
  embedding_dim: 50                           # Dimension of GloVe embeddings
  trainable: false                            # Freeze embeddings during training
  
# Encoder configuration
encoder:
  arch: avg_pool                              # Average pooling architecture
  hidden_dim: 128                             # Hidden layer dimension size
  tied_weights: true                          # Share weights between query and doc towers
  dropout: 0.1                                # Dropout rate for regularization
  
#==============================================================================
# Loss Function
#==============================================================================
loss:
  type: triplet                               # Triplet contrastive loss
  margin: 0.3                                 # Margin for triplet loss
  
#==============================================================================
# Training Parameters
#==============================================================================
batch_size: 128                               # Training batch size
epochs: 5                                     # Number of training epochs
optimizer:
  type: adam                                  # Adam optimizer
  lr: 0.0005                                  # Learning rate (lower for pretrained embeddings)
```

### 6. Create Example Scripts

#### Simple GloVe Search Example (examples/glove_search_example.py)

```python
"""
Simple example demonstrating the standalone GloVe search functionality.
"""

import logging
from twotower.utils import setup_logging
from twotower.glove_search import GloVeSearch

# Set up logging
setup_logging()
logger = logging.getLogger('examples.glove_search')

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
    search_engine = GloVeSearch(model_name='glove-wiki-gigaword-50')
    
    # Index the documents
    search_engine.index_documents(docs)
    
    # Perform search
    query = "I like cats."
    results = search_engine.search(query, top_k=3)
    
    # Display results
    print(f"Query: '{query}'")
    print("\nTop Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. '{result['document']}' (Score: {result['score']:.4f})")

if __name__ == "__main__":
    main()
```

#### Two-Tower GloVe Model Example (examples/two_tower_glove_example.py)

```python
"""
Example demonstrating the use of GloVe embeddings in the two-tower model.
"""

import logging
import mlx.core as mx
import torch
from twotower.utils import setup_logging, load_config
from twotower.tokenisers import build as build_tokeniser
from twotower.embeddings import build as build_embedding
from twotower.encoders import build_two_tower
from twotower.search import SemanticSearch

# Set up logging
setup_logging()
logger = logging.getLogger('examples.two_tower_glove')

def main():
    # Load configuration
    config = load_config('configs/glove_search.yml')
    
    # Build tokenizer
    tokenizer = build_tokeniser(config['tokeniser'])
    
    # Build embedding layer
    embedding = build_embedding(
        config['embedding']['type'],
        vocab_size=tokenizer.vocab_size,
        **config['embedding']
    )
    
    # Build two-tower model
    model = build_two_tower(
        embedding=embedding,
        **config['encoder']
    )
    
    # Sample documents
    docs = [
        "Machine learning is great for data analysis.",
        "Python programming makes handling data easy.",
        "Artificial intelligence and deep learning are popular today.",
        "Natural language processing allows computers to understand text.",
        "Dogs are very friendly animals."
    ]
    
    # Initialize semantic search
    search_engine = SemanticSearch(model, tokenizer)
    
    # Index documents
    search_engine.index_documents(docs)
    
    # Perform search
    query = "I like cats."
    results = search_engine.search(query, top_k=3)
    
    # Display results
    print(f"Query: '{query}'")
    print("\nTop Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. '{result['document']}' (Score: {result['score']:.4f})")

if __name__ == "__main__":
    main()
```

## Dependencies

Ensure the following dependencies are installed:

```
numpy
gensim
scikit-learn
torch
mlx
```

## Integration Steps

1. Update `requirements.txt` to include the new dependencies
2. Implement the `GloVeEmbedding` class in `twotower/embeddings.py`
3. Implement or update the average pooling encoder in `twotower/encoders.py`
4. Create the new search modules (`twotower/search.py` and `twotower/glove_search.py`)
5. Add the GloVe configuration file (`configs/glove_search.yml`)
6. Create example scripts to demonstrate functionality
7. Update documentation to reflect new features

## Testing Plan

1. Test standalone GloVe search utility with sample documents
2. Test GloVe embeddings integration with the two-tower model
3. Compare performance with existing embedding options
4. Measure memory usage and inference speed 
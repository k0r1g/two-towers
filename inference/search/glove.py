"""
GloVe-based semantic search utility.
"""

import numpy as np
import pickle
from typing import List, Dict, Any, Union
import logging
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseSearch

logger = logging.getLogger('inference.search.glove')

class GloVeSearch(BaseSearch):
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
    
    def index_documents(self, documents: List[str]) -> None:
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
        
    def save_index(self, filepath: str) -> None:
        """
        Save the document index to disk.
        
        Args:
            filepath: Path to save the index
        """
        if self.document_embeddings is None or self.documents is None:
            raise ValueError("No index to save. Call index_documents() first.")
            
        index_data = {
            'embeddings': self.document_embeddings,
            'documents': self.documents,
            'model_name': self.model.name if hasattr(self.model, 'name') else None
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)
            
        logger.info(f"Saved index with {len(self.documents)} documents to {filepath}")
        
    def load_index(self, filepath: str) -> None:
        """
        Load a document index from disk.
        
        Args:
            filepath: Path to the index file
        """
        with open(filepath, 'rb') as f:
            index_data = pickle.load(f)
            
        self.document_embeddings = index_data['embeddings']
        self.documents = index_data['documents']
        
        logger.info(f"Loaded index with {len(self.documents)} documents from {filepath}") 
"""
Base interface for all semantic search implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any

class BaseSearch(ABC):
    """
    Abstract base class for all semantic search implementations.
    All search classes should extend this interface.
    """
    
    @abstractmethod
    def index_documents(self, documents: List[str]) -> None:
        """
        Create and store embeddings for a list of documents.
        
        Args:
            documents: List of document strings to index
        """
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Union[str, float]]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: The search query string
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with document and similarity score
        """
        pass
    
    def save_index(self, filepath: str) -> None:
        """
        Save the document index to disk.
        
        Args:
            filepath: Path to save the index
        """
        raise NotImplementedError("This search implementation does not support saving indices")
    
    def load_index(self, filepath: str) -> None:
        """
        Load a document index from disk.
        
        Args:
            filepath: Path to the index file
        """
        raise NotImplementedError("This search implementation does not support loading indices") 
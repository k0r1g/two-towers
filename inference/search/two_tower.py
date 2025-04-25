"""
Two-Tower model based semantic search functionality.
"""

import torch
import pickle
import numpy as np
from typing import List, Dict, Union, Optional, Any
import logging

from .base import BaseSearch

logger = logging.getLogger('inference.search.two_tower')

class TwoTowerSearch(BaseSearch):
    """
    Semantic search using the query tower of a trained Two-Tower model.
    """
    def __init__(self, model, tokenizer, device='cpu'):
        """
        Initialize the search engine with a trained Two-Tower model.
        
        Args:
            model: A trained TwoTower model
            tokenizer: The tokenizer used for preprocessing text
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.document_embeddings = None
        self.documents = None
        
        # Move model to correct device
        self.model = self.model.to(self.device)
        
    def index_documents(self, documents: List[str]) -> None:
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
                encoded_batch = [
                    self.tokenizer.encode(doc) for doc in batch_docs
                ]
                padded_batch = [
                    self.tokenizer.truncate_and_pad(encoded, max_len=64) 
                    for encoded in encoded_batch
                ]
                inputs = torch.tensor(padded_batch, device=self.device)
                
                # Get document embeddings from the document tower
                doc_embeddings = self.model.document_tower(inputs)
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
        encoded_query = self.tokenizer.encode(query)
        padded_query = self.tokenizer.truncate_and_pad(encoded_query, max_len=64)
        query_input = torch.tensor([padded_query], device=self.device)
        
        # Get query embedding
        with torch.no_grad():
            query_embedding = self.model.query_tower(query_input)
        
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
        
    def save_index(self, filepath: str) -> None:
        """
        Save the document index to disk.
        
        Args:
            filepath: Path to save the index
        """
        if self.document_embeddings is None or self.documents is None:
            raise ValueError("No index to save. Call index_documents() first.")
            
        index_data = {
            'embeddings': self.document_embeddings.cpu().numpy() if torch.is_tensor(self.document_embeddings) else self.document_embeddings,
            'documents': self.documents,
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
            
        # Convert numpy arrays back to torch tensors if needed
        if isinstance(index_data['embeddings'], np.ndarray):
            self.document_embeddings = torch.tensor(index_data['embeddings'], device=self.device)
        else:
            self.document_embeddings = index_data['embeddings'].to(self.device)
            
        self.documents = index_data['documents']
        
        logger.info(f"Loaded index with {len(self.documents)} documents from {filepath}") 
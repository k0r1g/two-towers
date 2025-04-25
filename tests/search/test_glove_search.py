"""
Tests for the GloVe search utility.
"""

import unittest
import os
import tempfile
import numpy as np
from inference.search import GloVeSearch

class TestGloVeSearch(unittest.TestCase):
    """Test cases for GloVe-based semantic search."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.sample_docs = [
            "Machine learning is great for data analysis.",
            "Python programming makes handling data easy.",
            "Artificial intelligence and deep learning are popular today.",
            "Natural language processing allows computers to understand text.",
            "Dogs are very friendly animals."
        ]
        
        # Use a smaller model for faster testing
        self.search_engine = GloVeSearch(model_name='glove-twitter-25')
        
    def test_index_documents(self):
        """Test that documents are properly indexed."""
        self.search_engine.index_documents(self.sample_docs)
        
        # Check that document list is stored
        self.assertEqual(len(self.search_engine.documents), len(self.sample_docs))
        self.assertEqual(self.search_engine.documents, self.sample_docs)
        
        # Check that embeddings were created
        self.assertIsNotNone(self.search_engine.document_embeddings)
        self.assertEqual(self.search_engine.document_embeddings.shape, 
                        (len(self.sample_docs), self.search_engine.vector_size))
    
    def test_search_functionality(self):
        """Test basic search functionality."""
        self.search_engine.index_documents(self.sample_docs)
        
        # Test searching for machine learning related query
        ml_query = "Machine learning and AI"
        results = self.search_engine.search(ml_query, top_k=5)
        
        # Check result format
        self.assertEqual(len(results), len(self.sample_docs))  # top_k=5 with 5 docs should return all
        self.assertIn('document', results[0])
        self.assertIn('score', results[0])
        
        # The top result should be one of the AI/ML related documents
        ml_docs_indices = [0, 2, 3]  # Indices of ML-related documents
        top_doc = results[0]['document']
        self.assertIn(top_doc, [self.sample_docs[i] for i in ml_docs_indices])
        
        # Test searching for animal related query
        animal_query = "Pets and animals"
        results = self.search_engine.search(animal_query, top_k=5)
        
        # The top result should be the document about dogs
        self.assertEqual(results[0]['document'], self.sample_docs[4])
    
    def test_save_and_load_index(self):
        """Test saving and loading the search index."""
        self.search_engine.index_documents(self.sample_docs)
        
        # Create a temporary file for the index
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            # Save the index
            self.search_engine.save_index(temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # Create a new search engine
            new_search_engine = GloVeSearch(model_name='glove-twitter-25')
            
            # Load the index
            new_search_engine.load_index(temp_path)
            
            # Check that the loaded index has the same documents
            self.assertEqual(len(new_search_engine.documents), len(self.sample_docs))
            self.assertEqual(new_search_engine.documents, self.sample_docs)
            
            # Check that the embeddings were loaded
            self.assertTrue(np.array_equal(
                new_search_engine.document_embeddings,
                self.search_engine.document_embeddings
            ))
            
            # Test search with loaded index
            query = "Machine learning"
            orig_results = self.search_engine.search(query, top_k=1)
            loaded_results = new_search_engine.search(query, top_k=1)
            
            self.assertEqual(orig_results[0]['document'], loaded_results[0]['document'])
            self.assertAlmostEqual(orig_results[0]['score'], loaded_results[0]['score'])
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_different_glove_models(self):
        """Test using different GloVe models."""
        # Test with a different GloVe model
        other_model = 'glove-wiki-gigaword-50'
        search_engine_50d = GloVeSearch(model_name=other_model)
        
        # Index documents
        search_engine_50d.index_documents(self.sample_docs)
        
        # Check that embeddings have the correct dimension
        self.assertEqual(search_engine_50d.document_embeddings.shape[1], 50)
        
        # Test search functionality
        results = search_engine_50d.search("Machine learning", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIn('document', results[0])
        self.assertIn('score', results[0])

if __name__ == '__main__':
    unittest.main() 
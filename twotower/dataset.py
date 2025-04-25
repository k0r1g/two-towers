import torch
import pandas as pd
import logging
import collections
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Tuple, Sequence
from pathlib import Path

from .tokenisers import BaseTokeniser

# Set up logging
logger = logging.getLogger('twotower.dataset')

class TripletDataset(Dataset):
    """
    Dataset for training two-tower models with query, positive document, negative document triplets.
    """
    def __init__(
        self, 
        data_path: str, 
        tokeniser: BaseTokeniser,
        max_length: int = 64,
        load_to_memory: bool = True
    ):
        """
        Initialize the triplet dataset.
        
        Args:
            data_path: Path to the data file (parquet or tsv)
            tokeniser: Tokeniser instance for processing texts
            max_length: Maximum sequence length for tokenized inputs
            load_to_memory: Whether to load all data into memory (faster but uses more RAM)
        """
        self.data_path = data_path
        self.tokeniser = tokeniser
        self.max_length = max_length
        self.load_to_memory = load_to_memory
        
        logger.info(f"Loading dataset from {data_path}")
        
        # Load data based on file format
        self._load_data(data_path)
        
        # If tokeniser is not already fit, fit it on the texts
        if not hasattr(self.tokeniser, 'string_to_index') or not self.tokeniser.string_to_index:
            logger.info("Fitting tokeniser on dataset texts")
            all_texts = self.query_texts + self.positive_doc_texts + self.negative_doc_texts
            self.tokeniser.fit(all_texts)
        
        # Encode and pad the texts if loading to memory
        if load_to_memory:
            logger.info(f"Encoding texts with max_length={max_length}")
            self.encoded_queries = [self._encode_and_pad(query) for query in self.query_texts]
            self.encoded_positive_docs = [self._encode_and_pad(doc) for doc in self.positive_doc_texts]
            self.encoded_negative_docs = [self._encode_and_pad(doc) for doc in self.negative_doc_texts]
        
        # Log dataset statistics
        logger.info(f"Dataset loaded with {len(self)} triplets")
        if self.query_texts:
            query_sample = self.query_texts[0]
            pos_sample = self.positive_doc_texts[0][:50] + "..." if len(self.positive_doc_texts[0]) > 50 else self.positive_doc_texts[0]
            neg_sample = self.negative_doc_texts[0][:50] + "..." if len(self.negative_doc_texts[0]) > 50 else self.negative_doc_texts[0]
            logger.info(f"Sample query: '{query_sample}'")
            logger.info(f"Sample positive doc: '{pos_sample}'")
            logger.info(f"Sample negative doc: '{neg_sample}'")
    
    def _load_data(self, data_path: str):
        """
        Load data from file based on format (parquet or tsv).
        
        Args:
            data_path: Path to the data file
        """
        # Handle different file formats
        if data_path.endswith('.parquet'):
            self._load_from_parquet(data_path)
        elif data_path.endswith('.tsv'):
            self._load_from_tsv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}. Supported formats: .parquet, .tsv")
    
    def _load_from_parquet(self, data_path: str):
        """
        Load data from a parquet file.
        
        Args:
            data_path: Path to the parquet file
        """
        logger.info("Reading parquet format data")
        dataframe = pd.read_parquet(data_path)
        logger.info(f"Dataframe shape: {dataframe.shape}")
        logger.info(f"Dataframe columns: {dataframe.columns.tolist()}")
        
        # Check if the data is in triplets format or pairs format
        # Handle different column naming conventions
        query_col = None
        pos_col = None
        neg_col = None
        
        # Map possible column names
        query_columns = ['query', 'q_text']
        positive_columns = ['positive_doc', 'd_pos_text']
        negative_columns = ['negative_doc', 'd_neg_text']
        
        # Find which column names are present
        for col in query_columns:
            if col in dataframe.columns:
                query_col = col
                break
                
        for col in positive_columns:
            if col in dataframe.columns:
                pos_col = col
                break
                
        for col in negative_columns:
            if col in dataframe.columns:
                neg_col = col
                break
        
        # Check if the data is in triplets format 
        if query_col and pos_col and neg_col:
            # Already in triplets format
            logger.info(f"Data is already in triplets format with columns: {query_col}, {pos_col}, {neg_col}")
            self.query_texts = dataframe[query_col].tolist()
            self.positive_doc_texts = dataframe[pos_col].tolist()
            self.negative_doc_texts = dataframe[neg_col].tolist()
        elif all(col in dataframe.columns for col in ['query', 'document', 'label']):
            # Standard pairs format, need to convert to triplets
            logger.info("Data is in query-document-label format, converting to triplets")
            self._convert_pairs_to_triplets(
                queries=dataframe['query'].tolist(),
                documents=dataframe['document'].tolist(),
                labels=dataframe['label'].tolist()
            )
        else:
            # Unknown format
            raise ValueError(f"Unsupported dataframe format with columns: {dataframe.columns.tolist()}. "
                             f"Expected either triplets format with columns like 'query'/'q_text', 'positive_doc'/'d_pos_text', 'negative_doc'/'d_neg_text' "
                             f"or pairs format with columns 'query', 'document', 'label'")
    
    def _load_from_tsv(self, data_path: str):
        """
        Load data from a TSV file.
        
        Args:
            data_path: Path to the TSV file
        """
        logger.info("Reading TSV format data")
        
        # Try to import the dataset_factory for loading TSV
        try:
            # Import from dataset_factory
            from dataset_factory import load_synthetic_tsv
            
            # Check if the path is absolute
            data_path_obj = Path(data_path)
            if not data_path_obj.is_absolute():
                # Try to find it relative to the raw data directory
                try:
                    from dataset_factory.readers import RAW_DATA_DIR
                    data_path = RAW_DATA_DIR / data_path
                except ImportError:
                    logger.warning("Could not import RAW_DATA_DIR from dataset_factory. Using relative path as is.")
            
            dataframe = load_synthetic_tsv(data_path)
            logger.info(f"Loaded dataframe with shape: {dataframe.shape}")
            
            # Convert from pairs to triplets
            self._convert_pairs_to_triplets(
                queries=dataframe['query'].tolist(),
                documents=dataframe['document'].tolist(),
                labels=dataframe['label'].tolist()
            )
        except ImportError:
            logger.warning("dataset_factory module not available. Falling back to basic TSV loading.")
            
            # Basic TSV loading
            try:
                dataframe = pd.read_csv(data_path, sep='\t')
                if 'query' in dataframe.columns and 'document' in dataframe.columns and 'label' in dataframe.columns:
                    self._convert_pairs_to_triplets(
                        queries=dataframe['query'].tolist(),
                        documents=dataframe['document'].tolist(),
                        labels=dataframe['label'].tolist()
                    )
                else:
                    raise ValueError(f"TSV file must have 'query', 'document', and 'label' columns. Found: {dataframe.columns.tolist()}")
            except Exception as e:
                raise ValueError(f"Failed to load TSV file: {str(e)}")
    
    def _convert_pairs_to_triplets(self, queries: List[str], documents: List[str], labels: List[int]):
        """
        Convert query-document pairs with labels to query-positive-negative triplets.
        
        Args:
            queries: List of query texts
            documents: List of document texts
            labels: List of labels (1 for positive, 0 for negative)
        """
        logger.info("Grouping queries with positive and negative documents")
        
        # Group queries with positive and negative documents
        query_to_documents = collections.defaultdict(lambda: {'positive': [], 'negative': []})
        for query, document, label in zip(queries, documents, labels):
            if label == 1:
                query_to_documents[query]['positive'].append(document)
            else:
                query_to_documents[query]['negative'].append(document)
        
        logger.info(f"Created query-document mapping for {len(query_to_documents)} unique queries")
        
        # Log sample of query-document mapping
        sample_queries = list(query_to_documents.keys())[:3]
        for sample_query in sample_queries:
            pos_count = len(query_to_documents[sample_query]['positive'])
            neg_count = len(query_to_documents[sample_query]['negative'])
            logger.info(f"Query: '{sample_query}' has {pos_count} positive and {neg_count} negative documents")
            if pos_count > 0:
                logger.info(f"  Sample positive: '{query_to_documents[sample_query]['positive'][0][:50]}...'")
            if neg_count > 0:
                logger.info(f"  Sample negative: '{query_to_documents[sample_query]['negative'][0][:50]}...'")
        
        # Create triplets
        logger.info("Creating query-positive-negative triplets")
        triplets = []
        queries_with_both = 0
        
        for query, docs_dict in query_to_documents.items():
            if docs_dict['positive'] and docs_dict['negative']:  # Only keep queries with both positive and negative docs
                queries_with_both += 1
                for positive_doc in docs_dict['positive']:
                    for negative_doc in docs_dict['negative']:
                        triplets.append((query, positive_doc, negative_doc))
        
        logger.info(f"Created {len(triplets)} triplets from {queries_with_both}/{len(query_to_documents)} unique queries with both pos/neg docs")
        
        # Store triplets
        self.query_texts = [triplet[0] for triplet in triplets]
        self.positive_doc_texts = [triplet[1] for triplet in triplets]
        self.negative_doc_texts = [triplet[2] for triplet in triplets]
    
    def _encode_and_pad(self, text: str) -> List[int]:
        """
        Encode a text and pad/truncate to the specified maximum length.
        
        Args:
            text: Text to encode
        
        Returns:
            List of token IDs
        """
        return self.tokeniser.truncate_and_pad(
            self.tokeniser.encode(text), 
            self.max_length
        )
    
    def __len__(self) -> int:
        """Return the number of triplets in the dataset"""
        return len(self.query_texts)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a triplet by index.
        
        Args:
            index: Index of the triplet to retrieve
        
        Returns:
            Tuple of (query, positive_doc, negative_doc) tensors
        """
        if self.load_to_memory:
            # Use pre-encoded sequences
            return (
                torch.tensor(self.encoded_queries[index]),
                torch.tensor(self.encoded_positive_docs[index]),
                torch.tensor(self.encoded_negative_docs[index])
            )
        else:
            # Encode on-the-fly
            return (
                torch.tensor(self._encode_and_pad(self.query_texts[index])),
                torch.tensor(self._encode_and_pad(self.positive_doc_texts[index])),
                torch.tensor(self._encode_and_pad(self.negative_doc_texts[index]))
            )
    
    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size of the tokeniser"""
        return self.tokeniser.vocab_size

    def get_original_texts(self, index: int) -> Tuple[str, str, str]:
        """
        Get the original texts for a triplet.
        
        Args:
            index: Index of the triplet
        
        Returns:
            Tuple of (query_text, positive_doc_text, negative_doc_text)
        """
        return (
            self.query_texts[index],
            self.positive_doc_texts[index],
            self.negative_doc_texts[index]
        ) 
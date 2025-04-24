"""Dataset factory package for two-tower retrieval models."""

# Import key functions for easier access
from .readers import (
    get_ms_marco_dataset, 
    load_split, 
    save_dataset_as_parquet,
    load_synthetic_dataset,
    load_synthetic_tsv,
    convert_tsv_to_parquet,
    setup_data_dirs
)

from .synthetic_generators import (
    generate_synthetic_pairs,
    expand_synthetic_dataset
)

from .utils import (
    transform_and_save_dataset,
    convert_dataset_format,
    convert_triplets_to_pairs
)

# Export key classes and functions
__all__ = [
    # From readers
    'get_ms_marco_dataset',
    'load_split',
    'save_dataset_as_parquet',
    'load_synthetic_dataset',
    'load_synthetic_tsv',
    'convert_tsv_to_parquet',
    'setup_data_dirs',
    
    # From synthetic_generators
    'generate_synthetic_pairs',
    'expand_synthetic_dataset',
    
    # From utils
    'transform_and_save_dataset',
    'convert_dataset_format',
    'convert_triplets_to_pairs'
] 
"""
Two-Tower (Dual Encoder) neural network for document retrieval.

This package provides a modular implementation of the Two-Tower architecture
with configurable tokenisers, embeddings, encoder architectures, and loss functions.
"""

__version__ = "0.1.0"

# Import main components for easier access
from .tokenisers import CharTokeniser, build as build_tokeniser
from .embeddings import LookupEmbedding, FrozenWord2Vec, GloVeEmbedding, build as build_embedding
from .encoders import TwoTower, MeanPoolingTower, AveragePoolingTower, build_two_tower
from .losses import contrastive_triplet_loss, build as build_loss
from .dataset import TripletDataset
from .utils import (
    setup_logging, log_tensor_info, save_config, load_config,
    save_checkpoint, load_checkpoint, Timer
)

# Expose the train function
from .train import train_model

# Expose evaluation functionality
from .evaluate import evaluate_model, print_evaluation_results

# Expose HuggingFace Hub functionality
from .huggingface import (
    save_model_for_hub, upload_model_to_hub, load_model_from_hub,
    download_dataset_from_hub, save_and_upload
)

# Define what's available when using from twotower import *
__all__ = [
    # Core models
    'TwoTower', 'MeanPoolingTower', 'AveragePoolingTower',
    
    # Tokenisers
    'CharTokeniser', 'build_tokeniser',
    
    # Embeddings
    'LookupEmbedding', 'FrozenWord2Vec', 'GloVeEmbedding', 'build_embedding',
    
    # Model construction
    'build_two_tower',
    
    # Dataset
    'TripletDataset',
    
    # Loss functions
    'contrastive_triplet_loss', 'build_loss',
    
    # Training
    'train_model',
    
    # Evaluation
    'evaluate_model', 'print_evaluation_results',
    
    # HuggingFace Hub
    'save_model_for_hub', 'upload_model_to_hub', 'load_model_from_hub',
    'download_dataset_from_hub', 'save_and_upload',
    
    # Utilities
    'setup_logging', 'log_tensor_info', 'save_config', 'load_config',
    'save_checkpoint', 'load_checkpoint', 'Timer',
] 
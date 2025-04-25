#!/usr/bin/env python
"""
Train a two-tower model from a configuration file.

Usage:
    python -m twotower.train --config configs/char_tower.yml
"""

import argparse
import yaml
import torch
import wandb
import time
import os
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm

# Import components from other modules
from .tokenisers import build as build_tokeniser
from .embeddings import build as build_embedding
from .encoders import build_two_tower
from .dataset import TripletDataset
from .losses import build as build_loss
from .utils import setup_logging, log_tensor_info, save_config, load_config, save_checkpoint, Timer
from .huggingface import save_and_upload  # Import HuggingFace Hub integration

# Load default config from YAML file
try:
    # Try to load the default config file
    DEFAULT_CONFIG = load_config("configs/default_config.yml")
    
    # Extract constants from the config
    WANDB_PROJECT = DEFAULT_CONFIG["wandb"]["project"]
    WANDB_ENTITY = DEFAULT_CONFIG["wandb"]["entity"]
    DEFAULT_BATCH_SIZE = DEFAULT_CONFIG["batch_size"]
    DEFAULT_LEARNING_RATE = DEFAULT_CONFIG["learning_rate"]
    DEFAULT_EPOCHS = DEFAULT_CONFIG["epochs"]
    DEFAULT_EMBEDDING_DIM = DEFAULT_CONFIG["embedding"]["embedding_dim"]
    DEFAULT_HIDDEN_DIM = DEFAULT_CONFIG["encoder"]["hidden_dim"]
    CHECKPOINTS_DIR = DEFAULT_CONFIG["checkpoint_dir"]
    MAX_SEQUENCE_LENGTH = DEFAULT_CONFIG["max_sequence_length"]
except (FileNotFoundError, KeyError) as e:
    # Fallback defaults if config file not found or missing keys
    logger = logging.getLogger('twotower.train')
    logger.warning(f"Could not load default config: {str(e)}")
    logger.warning("Using fallback default values")
    
    WANDB_PROJECT = "two-tower-retrieval"
    WANDB_ENTITY = None
    DEFAULT_BATCH_SIZE = 256
    DEFAULT_LEARNING_RATE = 1e-3
    DEFAULT_EPOCHS = 3
    DEFAULT_EMBEDDING_DIM = 64
    DEFAULT_HIDDEN_DIM = 128
    CHECKPOINTS_DIR = "checkpoints"
    MAX_SEQUENCE_LENGTH = 64

# Get logger for this module
logger = logging.getLogger('twotower.train')

def train_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: callable,
    device: str,
    use_wandb: bool = False
) -> Dict[str, float]:
    """
    Train model for one epoch.
    
    Args:
        model: Two-tower model
        data_loader: DataLoader for training data
        optimizer: Optimizer for training
        loss_fn: Loss function
        device: Device to train on
        use_wandb: Whether to log metrics to wandb
    
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    total_loss = 0.0
    sample_count = 0
    
    # Track timing
    timer = Timer("Training")
    timer.start()
    
    # Metrics for tracking
    batch_times = []
    forward_times = []
    backward_times = []
    
    # Use tqdm for progress bar
    progress_bar = tqdm(data_loader, desc=f'Training', ncols=100)
    
    for batch_idx, (queries, positive_docs, negative_docs) in enumerate(progress_bar):
        batch_start_time = time.time()
        
        # Move data to device
        queries = queries.to(device)
        positive_docs = positive_docs.to(device)
        negative_docs = negative_docs.to(device)
        
        # Log tensor shapes for first batch
        if batch_idx == 0:
            logger.debug(f"Batch {batch_idx} tensor shapes:")
            log_tensor_info(queries, "queries")
            log_tensor_info(positive_docs, "positive_docs")
            log_tensor_info(negative_docs, "negative_docs")
        
        # Forward pass
        forward_start = time.time()
        query_vectors, positive_doc_vectors, negative_doc_vectors = model(
            queries, positive_docs, negative_docs
        )
        forward_time = time.time() - forward_start
        forward_times.append(forward_time)
        
        # Log tensor shapes for first batch
        if batch_idx == 0:
            log_tensor_info(query_vectors, "query_vectors")
            log_tensor_info(positive_doc_vectors, "positive_doc_vectors")
            log_tensor_info(negative_doc_vectors, "negative_doc_vectors")
        
        # Compute loss
        loss = loss_fn(query_vectors, positive_doc_vectors, negative_doc_vectors)
        
        # Backward pass and optimization
        backward_start = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        backward_time = time.time() - backward_start
        backward_times.append(backward_time)
        
        # Calculate similarity scores for monitoring
        with torch.no_grad():
            pos_similarity = torch.nn.functional.cosine_similarity(
                query_vectors, positive_doc_vectors
            ).mean().item()
            neg_similarity = torch.nn.functional.cosine_similarity(
                query_vectors, negative_doc_vectors
            ).mean().item()
            similarity_diff = pos_similarity - neg_similarity
        
        # Update totals
        batch_loss = loss.item()
        total_loss += batch_loss * len(queries)
        sample_count += len(queries)
        
        # Calculate batch time
        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'pos_sim': f'{pos_similarity:.4f}',
            'neg_sim': f'{neg_similarity:.4f}',
            'diff': f'{similarity_diff:.4f}'
        })
        
        # Log metrics to wandb if enabled
        if use_wandb:
            wandb.log({
                'train/batch': batch_idx,
                'train/batch_loss': batch_loss,
                'train/pos_similarity': pos_similarity,
                'train/neg_similarity': neg_similarity,
                'train/similarity_diff': similarity_diff,
                'performance/batch_time': batch_time,
                'performance/forward_time': forward_time,
                'performance/backward_time': backward_time,
                'performance/samples_per_second': len(queries) / batch_time,
            })
            
            # Compute and log gradient norms every 10 batches
            if batch_idx % 10 == 0:
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                
                wandb.log({
                    'train/batch': batch_idx,
                    'gradients/total_norm': total_norm,
                    'train/grad_norm': total_norm  # Add duplicate with preferred metric name
                })
    
    # Calculate epoch metrics
    epoch_loss = total_loss / sample_count if sample_count > 0 else float('inf')
    epoch_time = timer.stop()
    
    # Average metrics
    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
    avg_forward_time = sum(forward_times) / len(forward_times) if forward_times else 0
    avg_backward_time = sum(backward_times) / len(backward_times) if backward_times else 0
    
    metrics = {
        'loss': epoch_loss,
        'time': epoch_time,
        'avg_batch_time': avg_batch_time,
        'avg_forward_time': avg_forward_time,
        'avg_backward_time': avg_backward_time,
    }
    
    logger.info(f"Training completed in {epoch_time:.2f}s")
    logger.info(f"  Mean loss: {epoch_loss:.6f}")
    logger.info(f"  Average batch time: {avg_batch_time:.4f}s (forward: {avg_forward_time:.4f}s, backward: {avg_backward_time:.4f}s)")
    
    return metrics

def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    loss_fn: callable,
    device: str
) -> Dict[str, float]:
    """
    Evaluate model on a validation dataset.
    
    Args:
        model: Two-tower model
        data_loader: DataLoader for validation data
        loss_fn: Loss function
        device: Device to evaluate on
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    total_loss = 0.0
    total_pos_sim = 0.0
    total_neg_sim = 0.0
    total_diff = 0.0
    sample_count = 0
    
    logger.info("Starting evaluation")
    
    with torch.no_grad():
        for queries, positive_docs, negative_docs in tqdm(data_loader, desc="Evaluating"):
            # Move data to device
            queries = queries.to(device)
            positive_docs = positive_docs.to(device)
            negative_docs = negative_docs.to(device)
            
            # Forward pass
            query_vectors, positive_doc_vectors, negative_doc_vectors = model(
                queries, positive_docs, negative_docs
            )
            
            # Compute loss
            loss = loss_fn(query_vectors, positive_doc_vectors, negative_doc_vectors)
            
            # Calculate similarity scores
            pos_similarity = torch.nn.functional.cosine_similarity(
                query_vectors, positive_doc_vectors
            ).mean().item()
            neg_similarity = torch.nn.functional.cosine_similarity(
                query_vectors, negative_doc_vectors
            ).mean().item()
            similarity_diff = pos_similarity - neg_similarity
            
            # Update totals
            batch_size = len(queries)
            total_loss += loss.item() * batch_size
            total_pos_sim += pos_similarity * batch_size
            total_neg_sim += neg_similarity * batch_size
            total_diff += similarity_diff * batch_size
            sample_count += batch_size
    
    # Calculate average metrics
    metrics = {
        'loss': total_loss / sample_count if sample_count > 0 else float('inf'),
        'pos_similarity': total_pos_sim / sample_count if sample_count > 0 else 0,
        'neg_similarity': total_neg_sim / sample_count if sample_count > 0 else 0,
        'similarity_diff': total_diff / sample_count if sample_count > 0 else 0,
    }
    
    logger.info(f"Evaluation results:")
    logger.info(f"  Loss: {metrics['loss']:.6f}")
    logger.info(f"  Positive similarity: {metrics['pos_similarity']:.6f}")
    logger.info(f"  Negative similarity: {metrics['neg_similarity']:.6f}")
    logger.info(f"  Similarity difference: {metrics['similarity_diff']:.6f}")
    
    return metrics

def build_pipeline(config: Dict[str, Any], device: str) -> Tuple:
    """
    Build the complete training pipeline from configuration.
    
    Args:
        config: Configuration dictionary
        device: Device to use
    
    Returns:
        Tuple of (model, dataset, optimizer, loss_fn)
    """
    # Stage 1: Tokeniser
    tokeniser_config = config.get('tokeniser', {'type': 'char', 'max_len': MAX_SEQUENCE_LENGTH})
    tokeniser = build_tokeniser(tokeniser_config.get('type', 'char'))
    
    # Stage 2: Dataset
    dataset_config = config.get('dataset', {})
    max_length = tokeniser_config.get('max_len', MAX_SEQUENCE_LENGTH)
    
    dataset = TripletDataset(
        data_path=config['data'],
        tokeniser=tokeniser,
        max_length=max_length,
        load_to_memory=dataset_config.get('load_to_memory', True)
    )
    
    # Stage 3: Embedding
    embedding_config = config.get('embedding', {'type': 'lookup', 'embedding_dim': DEFAULT_EMBEDDING_DIM})
    embedding_type = embedding_config.get('type', 'lookup')
    embedding_dim = embedding_config.get('embedding_dim', DEFAULT_EMBEDDING_DIM)
    
    embedding = build_embedding(
        name=embedding_type,
        vocab_size=dataset.vocab_size,
        embedding_dim=embedding_dim
    )
    
    # Stage 4: Encoder / Tower
    encoder_config = config.get('encoder', {'arch': 'mean', 'hidden_dim': DEFAULT_HIDDEN_DIM})
    hidden_dim = encoder_config.get('hidden_dim', DEFAULT_HIDDEN_DIM)
    tied_weights = encoder_config.get('tied_weights', False)
    
    model = build_two_tower(
        tower_name=encoder_config.get('arch', 'mean'),
        embedding=embedding,
        hidden_dim=hidden_dim,
        tied_weights=tied_weights
    ).to(device)
    
    # Stage 5: Loss function
    loss_config = config.get('loss', {'type': 'triplet', 'margin': 0.2})
    loss_type = loss_config.get('type', 'triplet')
    loss_kwargs = {k: v for k, v in loss_config.items() if k != 'type'}
    
    loss_fn = build_loss(loss_type, **loss_kwargs)
    
    # Initialize optimizer
    optimizer_config = config.get('optimizer', {'type': 'adamw', 'lr': DEFAULT_LEARNING_RATE})
    lr = optimizer_config.get('lr', DEFAULT_LEARNING_RATE)
    
    if optimizer_config.get('type', 'adamw').lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_config.get('type', '').lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_config.get('type', '').lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=lr,
            momentum=optimizer_config.get('momentum', 0.9)
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    return model, dataset, optimizer, loss_fn

def train_model(config: Dict[str, Any]) -> torch.nn.Module:
    """
    Train a two-tower model using the provided configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Trained model
    """
    # Extract training parameters
    epochs = config.get('epochs', DEFAULT_EPOCHS)
    batch_size = config.get('batch_size', DEFAULT_BATCH_SIZE)
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = config.get('checkpoint_dir', CHECKPOINTS_DIR)
    use_wandb = config.get('use_wandb', False)
    
    # Extract HuggingFace Hub parameters
    hf_config = config.get('huggingface', {})
    push_to_hub = hf_config.get('push_to_hub', False)
    hf_repo_id = hf_config.get('repo_id', 'mlx7-two-tower')
    hf_private = hf_config.get('private', False)
    
    # Initialize wandb if enabled
    if use_wandb:
        wandb_config = config.get('wandb', {})
        wandb.init(
            project=wandb_config.get('project', WANDB_PROJECT),
            entity=wandb_config.get('entity', WANDB_ENTITY),
            name=wandb_config.get('run_name'),
            config=config
        )
        logger.info(f"Initialized W&B run: {wandb.run.name}")
    
    # Build the pipeline components
    model, dataset, optimizer, loss_fn = build_pipeline(config, device)
    
    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.get('num_workers', 0),
        pin_memory=(device == 'cuda')
    )
    
    # Track best model
    best_loss = float('inf')
    best_model_path = None
    
    # Training loop
    logger.info(f"Starting training for {epochs} epochs")
    
    for epoch in range(1, epochs + 1):
        logger.info(f"Epoch {epoch}/{epochs}")
        
        # Train for one epoch
        train_metrics = train_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            use_wandb=use_wandb
        )
        
        # Log epoch metrics
        epoch_loss = train_metrics['loss']
        
        # Log to wandb if enabled
        if use_wandb:
            # Get current learning rate from optimizer
            current_lr = optimizer.param_groups[0]['lr']
            
            wandb.log({
                'epoch': epoch,
                'train/epoch_loss': epoch_loss,
                'train/epoch_time': train_metrics['time'],
                'train/learning_rate': current_lr,
                'train/batch_size': batch_size
            })
        
        # Save model if it's the best so far
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            logger.info(f"New best model with loss: {best_loss:.6f}")
            
            # Save checkpoint
            best_model_path = save_checkpoint(
                model=model,
                tokeniser_vocab=dataset.tokeniser.string_to_index,
                optimizer=optimizer,
                epoch=epoch,
                loss=epoch_loss,
                checkpoint_dir=checkpoint_dir,
                save_best=True
            )
    
    logger.info(f"Training completed. Best loss: {best_loss:.6f}")
    
    # Push model to HuggingFace Hub if enabled
    if push_to_hub and best_model_path:
        logger.info(f"Pushing model to HuggingFace Hub: {hf_repo_id}")
        try:
            # Save and upload model to HuggingFace Hub
            repo_url = save_and_upload(
                model=model,
                tokenizer=dataset.tokeniser,
                config=config,
                repo_id=hf_repo_id,
                local_dir=os.path.join(checkpoint_dir, "hub_export"),
                private=hf_private
            )
            logger.info(f"Model successfully pushed to HuggingFace Hub: {repo_url}")
            
            # Log the HuggingFace Hub URL to W&B if enabled
            if use_wandb:
                wandb.log({"huggingface_hub_url": repo_url})
        except Exception as e:
            logger.error(f"Failed to push model to HuggingFace Hub: {str(e)}")
    
    # Close wandb run if enabled
    if use_wandb:
        wandb.finish()
    
    return model

def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="Train a two-tower model")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', 
                     help="Logging level")
    parser.add_argument("--log_file", default=None, help="Path to log file")
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
    
    # Add HuggingFace Hub arguments
    parser.add_argument("--push_to_hub", action="store_true", help="Push trained model to HuggingFace Hub")
    parser.add_argument("--hub_repo_id", default="mlx7-two-tower", help="HuggingFace Hub repository ID")
    parser.add_argument("--hub_private", action="store_true", help="Make HuggingFace Hub repository private")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(
        log_level=args.log_level,
        log_file=args.log_file
    )
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.use_wandb:
        config['use_wandb'] = True
    
    # Override HuggingFace Hub config with command line arguments
    if args.push_to_hub:
        if 'huggingface' not in config:
            config['huggingface'] = {}
        config['huggingface']['push_to_hub'] = True
        config['huggingface']['repo_id'] = args.hub_repo_id
        config['huggingface']['private'] = args.hub_private
    
    # Train model
    model = train_model(config)
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main() 
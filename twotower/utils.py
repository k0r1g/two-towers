import logging
import torch
import os
import json
import yaml
import time
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple

# Set up package-wide logger
logger = logging.getLogger('twotower')

def setup_logging(
    log_level: str = 'INFO',
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (if None, only console logging is used)
        console: Whether to enable console output
    
    Returns:
        Configured logger
    """
    # Convert log level string to actual level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Configure root logger
    logger.setLevel(numeric_level)
    
    # Clear existing handlers to avoid duplicate logs
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.info(f"Logging configured with level={log_level}, file={log_file}, console={console}")
    return logger

def log_tensor_info(tensor, name="tensor") -> None:
    """
    Log helpful information about tensors for debugging.
    
    Args:
        tensor: Tensor or list to log information about
        name: Name of the tensor for logging
    """
    if isinstance(tensor, torch.Tensor):
        logger.info(f"{name} shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")
        logger.info(f"{name} stats: min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, "
                    f"mean={tensor.float().mean().item():.4f}, std={tensor.float().std().item():.4f}")
        if tensor.numel() < 10:
            logger.info(f"{name} full content: {tensor}")
        else:
            logger.info(f"{name} sample: {tensor.flatten()[:5].tolist()} ... {tensor.flatten()[-5:].tolist()}")
    elif isinstance(tensor, list):
        logger.info(f"{name} type: list, length: {len(tensor)}")
        if len(tensor) < 10:
            logger.info(f"{name} full content: {tensor}")
        else:
            logger.info(f"{name} sample: {tensor[:3]} ... {tensor[-3:]}")
    else:
        logger.info(f"{name}: {tensor}")

def save_config(config: Dict[str, Any], path: str) -> None:
    """
    Save a configuration dictionary to a YAML file.
    
    Args:
        config: Configuration dictionary
        path: Path to save the configuration file
    """
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Configuration saved to {path}")

def load_config(path: str) -> Dict[str, Any]:
    """
    Load a configuration dictionary from a YAML file.
    
    Args:
        path: Path to the configuration file
    
    Returns:
        Configuration dictionary
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded from {path}")
    return config

def save_checkpoint(
    model: torch.nn.Module,
    tokeniser_vocab: Dict[str, int],
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    loss: float = float('inf'),
    checkpoint_dir: str = 'checkpoints',
    checkpoint_name: Optional[str] = None,
    save_best: bool = True
) -> str:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        tokeniser_vocab: Tokeniser vocabulary (string_to_index)
        optimizer: PyTorch optimizer (optional)
        epoch: Current epoch number
        loss: Current loss value
        checkpoint_dir: Directory to save checkpoints
        checkpoint_name: Name for the checkpoint file (if None, creates a timestamped name)
        save_best: Whether to also save a copy as best_model.pt
    
    Returns:
        Path to the saved checkpoint
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create timestamp for checkpoint name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use provided name or create one with timestamp
    if checkpoint_name is None:
        checkpoint_name = f"two_tower_{timestamp}_epoch{epoch}.pt"
    
    # Create checkpoint path
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    # Create checkpoint data dictionary
    checkpoint = {
        'model': model.state_dict(),
        'vocab': tokeniser_vocab,
        'epoch': epoch,
        'loss': loss,
        'timestamp': timestamp
    }
    
    # Add optimizer state if provided
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save as best_model.pt if requested
    if save_best:
        best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
        torch.save(checkpoint, best_model_path)
        logger.info(f"Saved best model to {best_model_path}")
    
    return checkpoint_path

def load_checkpoint(
    checkpoint_path: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: PyTorch model to load weights into (optional)
        optimizer: PyTorch optimizer to load state into (optional)
        device: Device to load tensors to
    
    Returns:
        Dictionary with checkpoint data
    """
    # Load checkpoint data
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Log checkpoint info
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    logger.info(f"Checkpoint from epoch {checkpoint.get('epoch', 'unknown')} with loss {checkpoint.get('loss', 'unknown')}")
    
    # Load model weights if model is provided
    if model is not None and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        logger.info(f"Loaded model weights from checkpoint")
    
    # Load optimizer state if optimizer is provided
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(f"Loaded optimizer state from checkpoint")
    
    return checkpoint

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
    
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

class Timer:
    """Simple timer for measuring elapsed time."""
    
    def __init__(self, name: str = "Timer"):
        """
        Initialize timer.
        
        Args:
            name: Name for the timer (used in logging)
        """
        self.name = name
        self.start_time = None
        self.splits = []
    
    def start(self) -> float:
        """Start the timer."""
        self.start_time = time.time()
        self.splits = []
        logger.debug(f"{self.name} started")
        return self.start_time
    
    def split(self, split_name: str = None) -> float:
        """
        Record a split time.
        
        Args:
            split_name: Name for this split
        
        Returns:
            Time elapsed since last split (or start)
        """
        if self.start_time is None:
            self.start()
            return 0.0
        
        current = time.time()
        last_time = self.start_time if not self.splits else self.splits[-1][1]
        elapsed = current - last_time
        
        split_info = (split_name or f"Split {len(self.splits)+1}", current, elapsed)
        self.splits.append(split_info)
        
        logger.debug(f"{self.name} - {split_info[0]}: {elapsed:.4f}s")
        return elapsed
    
    def stop(self) -> float:
        """
        Stop the timer and return total elapsed time.
        
        Returns:
            Total elapsed time
        """
        if self.start_time is None:
            return 0.0
        
        end_time = time.time()
        total_time = end_time - self.start_time
        
        logger.debug(f"{self.name} stopped after {total_time:.4f}s")
        return total_time
    
    def summary(self) -> Dict[str, float]:
        """
        Generate a summary of timer results.
        
        Returns:
            Dictionary with timing information
        """
        if self.start_time is None:
            return {"error": "Timer not started"}
        
        current = time.time()
        total_time = current - self.start_time
        
        result = {
            "total_time": total_time,
            "splits": {s[0]: s[2] for s in self.splits},
            "split_percentages": {s[0]: (s[2] / total_time) * 100 for s in self.splits}
        }
        
        # Log summary
        logger.info(f"{self.name} summary:")
        logger.info(f"  Total time: {total_time:.4f}s")
        for name, elapsed in result["splits"].items():
            percentage = result["split_percentages"][name]
            logger.info(f"  {name}: {elapsed:.4f}s ({percentage:.1f}%)")
        
        return result 
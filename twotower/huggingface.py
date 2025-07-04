"""
HuggingFace Hub integration for the Two-Tower model.

This module provides functions to save, load, and share Two-Tower models
using the Hugging Face Hub.
"""

import os
import json
import torch
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
import logging

# Import HF Hub utilities
from huggingface_hub import (
    HfApi, hf_hub_download, snapshot_download, 
    create_repo, upload_file, upload_folder
)

# Import local utilities
from tools.huggingface import setup_repository


def save_model_for_hub(
    model: torch.nn.Module,
    tokenizer: Any,
    config: Dict[str, Any],
    output_dir: str,
    model_name: str = "best_model.pt",
    config_name: str = "config.yml",
    tokenizer_name: str = "tokenizer.json"
) -> str:
    """
    Save a model, tokenizer, and config for uploading to the Hugging Face Hub.
    
    Args:
        model: PyTorch model
        tokenizer: Tokenizer object
        config: Model configuration
        output_dir: Directory to save files
        model_name: Name for the model file
        config_name: Name for the config file
        tokenizer_name: Name for the tokenizer file
        
    Returns:
        Path to the output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, model_name)
    torch.save({"model": model.state_dict()}, model_path)
    
    # Save config
    config_path = os.path.join(output_dir, config_name)
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    # Save tokenizer (assuming it has a to_dict method, adapt as needed)
    tokenizer_path = os.path.join(output_dir, tokenizer_name)
    tokenizer_dict = tokenizer.t2i if hasattr(tokenizer, "t2i") else None
    if tokenizer_dict:
        with open(tokenizer_path, "w") as f:
            json.dump(tokenizer_dict, f)
    
    return output_dir


def upload_model_to_hub(
    repo_id: str,
    local_dir: str,
    commit_message: str = "Upload Two-Tower model",
    token: Optional[str] = None,
    private: bool = False,
    create_if_missing: bool = True
) -> str:
    """
    Upload a saved Two-Tower model to the Hugging Face Hub.
    
    Args:
        repo_id: Repository ID (username/repo_name)
        local_dir: Local directory containing the model files
        commit_message: Commit message
        token: Hugging Face token
        private: Whether to create a private repository if it doesn't exist
        create_if_missing: Whether to create the repository if it doesn't exist
        
    Returns:
        URL of the repository
    """
    logger = logging.getLogger("twotower")
    
    # Get token from environment if not provided
    if token is None:
        token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
        if token:
            logger.info("Using HUGGINGFACE_ACCESS_TOKEN from environment for upload")
        else:
            logger.warning("No token provided and HUGGINGFACE_ACCESS_TOKEN not found in environment. "
                         "Authentication may fail during upload.")
    
    # Check if local directory exists
    if not os.path.exists(local_dir):
        raise ValueError(f"Local directory '{local_dir}' does not exist.")

    # Initialize API with token
    api = HfApi(token=token)
    
    try:
        # Verify token works by checking user info
        user_info = api.whoami()
        username = user_info["name"]
        logger.info(f"Authenticated as: {username}")
    except Exception as e:
        logger.error(f"Authentication failed: {str(e)}")
        raise ValueError("Failed to authenticate with Hugging Face Hub. Please check your token.")
    
    # Create repository if it doesn't exist
    if create_if_missing:
        try:
            repo_url = setup_repository(
                repo_id=repo_id,
                private=private,
                token=token,
                exist_ok=True,
                repo_type="model"
            )
            logger.info(f"Repository created or already exists: {repo_id}")
        except Exception as e:
            logger.error(f"Failed to create repository: {str(e)}")
            raise
    
    # List files to upload
    file_count = sum(1 for _ in Path(local_dir).glob('**/*') if _.is_file())
    logger.info(f"Uploading {file_count} files to {repo_id}...")
    
    # Upload the directory
    try:
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            commit_message=commit_message,
            token=token
        )
        logger.info(f"Upload completed successfully")
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise
    
    repo_url = f"https://huggingface.co/{repo_id}"
    logger.info(f"Model uploaded to {repo_url}")
    return repo_url


def load_model_from_hub(
    repo_id: str,
    model_class: Any,
    tokenizer_class: Any,
    model_filename: str = "best_model.pt",
    config_filename: str = "config.yml",
    tokenizer_filename: str = "tokenizer.json",
    token: Optional[str] = None,
    revision: Optional[str] = None
) -> Tuple[torch.nn.Module, Any, Dict[str, Any]]:
    """
    Load a Two-Tower model from the Hugging Face Hub.
    
    Args:
        repo_id: Repository ID (username/repo_name)
        model_class: Model class to instantiate
        tokenizer_class: Tokenizer class to instantiate
        model_filename: Name of the model file in the repository
        config_filename: Name of the config file in the repository
        tokenizer_filename: Name of the tokenizer file in the repository
        token: Hugging Face token
        revision: Git revision to use (branch, tag, or commit hash)
        
    Returns:
        Tuple of (model, tokenizer, config)
    """
    # Download files from Hub
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=model_filename,
        token=token,
        revision=revision
    )
    
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename=config_filename,
        token=token,
        revision=revision
    )
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Load tokenizer
    try:
        tokenizer_path = hf_hub_download(
            repo_id=repo_id,
            filename=tokenizer_filename,
            token=token,
            revision=revision
        )
        with open(tokenizer_path, "r") as f:
            tokenizer_dict = json.load(f)
        
        # Initialize tokenizer (adjust based on your tokenizer's __init__ method)
        tokenizer = tokenizer_class()
        tokenizer.t2i = tokenizer_dict
        tokenizer.i2t = {v: k for k, v in tokenizer_dict.items()}
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")
        tokenizer = None
    
    # Initialize model (adjust based on your model's __init__ method)
    model = model_class(**config.get("encoder", {}))
    
    # Load state dict
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    
    return model, tokenizer, config


def download_dataset_from_hub(
    repo_id: str,
    output_dir: str,
    filename: Optional[str] = None,
    token: Optional[str] = None,
    revision: Optional[str] = None
) -> str:
    """
    Download a dataset from the Hugging Face Hub.
    
    Args:
        repo_id: Repository ID (username/repo_name)
        output_dir: Directory to save the dataset
        filename: Specific file to download (if None, downloads the entire repository)
        token: Hugging Face token
        revision: Git revision to use (branch, tag, or commit hash)
        
    Returns:
        Path to the downloaded dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if filename:
        # Download a specific file
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
            revision=revision,
            local_dir=output_dir
        )
        return file_path
    else:
        # Download the entire repository
        local_dir = snapshot_download(
            repo_id=repo_id,
            token=token,
            revision=revision,
            local_dir=output_dir
        )
        return local_dir


# Simplified interfaces for common operations

def save_and_upload(
    model: torch.nn.Module,
    tokenizer: Any,
    config: Dict[str, Any],
    repo_id: str = "mlx7-two-tower",
    local_dir: str = "hub_export",
    private: bool = False,
    token: Optional[str] = None,
    force_upload: bool = False
) -> str:
    """
    Save and upload a Two-Tower model to the Hugging Face Hub in one go.
    
    Args:
        model: PyTorch model
        tokenizer: Tokenizer object
        config: Model configuration
        repo_id: Repository ID 
        local_dir: Temporary directory to save files
        private: Whether to create a private repository
        token: Hugging Face token (if None, will look for HUGGINGFACE_ACCESS_TOKEN in environment)
        force_upload: Force upload even if there are issues
        
    Returns:
        URL of the repository
    """
    logger = logging.getLogger("twotower")
    
    # Get token from environment if not provided
    if token is None:
        token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
        if token:
            logger.info("Using HUGGINGFACE_ACCESS_TOKEN from environment")
        else:
            logger.warning("No token provided and HUGGINGFACE_ACCESS_TOKEN not found in environment. "
                         "Authentication may fail.")
    
    # Save the model locally
    try:
        save_dir = save_model_for_hub(model, tokenizer, config, local_dir)
        logger.info(f"Model saved locally to {save_dir}")
    except Exception as e:
        logger.error(f"Error saving model locally: {str(e)}")
        if not force_upload:
            raise
    
    # Upload to the Hub
    try:
        logger.info(f"Uploading model to {repo_id}...")
        repo_url = upload_model_to_hub(
            repo_id=repo_id, 
            local_dir=save_dir,
            token=token,
            private=private,
            create_if_missing=True
        )
        logger.info(f"Model successfully uploaded to {repo_url}")
        return repo_url
    except Exception as e:
        logger.error(f"Error uploading model to Hub: {str(e)}")
        if not force_upload:
            raise
        return f"https://huggingface.co/{repo_id} (upload may have failed)" 
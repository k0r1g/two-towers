#!/usr/bin/env python
"""
Train a two-tower model on the MS MARCO dataset.

This script demonstrates a full training pipeline:
1. Download MS MARCO dataset if not already available
2. Convert to triplets format if needed
3. Sample the dataset to the specified size
4. Train the two-tower model

Usage:
    # Train on 10,000 samples from MS MARCO
    python train_with_msmarco.py --samples 10000 --epochs 3
    
    # Skip the data preparation step
    python train_with_msmarco.py --skip_prepare --samples 5000
"""

import argparse
import subprocess
import os
import random
from pathlib import Path
import logging
import pandas as pd
import yaml

# Import from dataset_factory
from dataset_factory import (
    get_ms_marco_dataset,
    save_dataset_as_parquet,
    transform_and_save_dataset
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('train_with_msmarco')

def find_preset_file(preset_path):
    """Find a preset file, trying common variations if the exact path doesn't exist."""
    # First check if the exact path exists
    if os.path.exists(preset_path):
        return preset_path
    
    # Get the directory and filename
    preset_dir = os.path.dirname(preset_path) or '.'
    preset_name = os.path.basename(preset_path)
    
    # Remove the extension if it exists
    preset_base = os.path.splitext(preset_name)[0]
    
    # Try common variations
    variations = [
        f"{preset_base}.yml",
        f"{preset_base}.yaml",
        f"{preset_base}_preset.yml",
        f"{preset_base}_preset.yaml"
    ]
    
    # Check for other files in the directory with similar names
    if os.path.exists(preset_dir):
        for filename in os.listdir(preset_dir):
            if filename.startswith(preset_base) and filename.endswith(('.yml', '.yaml')):
                full_path = os.path.join(preset_dir, filename)
                logger.info(f"Found possible preset file match: {full_path}")
                return full_path
    
    # Try the variations
    for var in variations:
        var_path = os.path.join(preset_dir, var)
        if os.path.exists(var_path):
            logger.info(f"Found preset file at {var_path}")
            return var_path
    
    # If we get here, none of the variations exists either
    logger.warning(f"Could not find preset file at {preset_path} or any common variations")
    return preset_path  # Return the original path for consistent error messaging

def main():
    parser = argparse.ArgumentParser(description="Train with MS MARCO dataset")
    # Data preparation options
    parser.add_argument('--force_download', action='store_true', help='Force redownload of MS MARCO dataset')
    parser.add_argument('--skip_prepare', action='store_true', help='Skip data preparation steps')
    parser.add_argument('--split', default='train', choices=['train', 'dev', 'eval'], help='Dataset split to use')
    parser.add_argument('--preset', default='presets/multi_pos_multi_neg.yml', 
                        help='Preset configuration for build_dataset')
    
    # Sampling options
    parser.add_argument('--samples', type=int, default=None, 
                        help='Number of triplets to sample (default: use all available data)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    
    # Training options
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--skip_training', action='store_true', help='Skip model training step')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    
    args = parser.parse_args()

    # Find the preset file, looking for common variations if needed
    preset_path = find_preset_file(args.preset)
    
    # Verify the preset file exists and is valid
    if not os.path.exists(preset_path):
        logger.error(f"Preset file {preset_path} not found. Please specify a valid preset file.")
        return
    
    try:
        # Validate that the preset file is a valid YAML file
        with open(preset_path, 'r') as f:
            preset_config = yaml.safe_load(f)
            
        # Basic validation
        required_keys = ['positive_selector', 'negative_sampler', 'negatives_per_pos']
        missing_keys = [key for key in required_keys if key not in preset_config]
        if missing_keys:
            logger.warning(f"Preset file {preset_path} is missing required keys: {missing_keys}")
    except Exception as e:
        logger.warning(f"Error validating preset file {preset_path}: {str(e)}")
        
    # Get the preset name without directory or extension for file naming
    preset_name = Path(preset_path).stem
    
    # Define file paths
    triplets_file = f"{args.split}_{preset_name}.parquet"
    triplets_parquet = Path(f"data/processed/{triplets_file}")
    
    if args.samples is not None:
        sample_triplets_file = f"{args.split}_{preset_name}_sample_{args.samples}.parquet"
        sample_triplets_parquet = Path(f"data/processed/{sample_triplets_file}")
    else:
        sample_triplets_parquet = triplets_parquet

    # Step 1: Prepare the MS MARCO dataset
    if not args.skip_prepare:
        logger.info("Preparing MS MARCO dataset...")
        
        # Step 1.1: Download the dataset if needed
        dataset = get_ms_marco_dataset(force_download=args.force_download)
        
        # Step 1.2: Save as parquet files
        parquet_files = save_dataset_as_parquet(dataset, force_save=args.force_download)
        
        # Step 1.3: Create the triplets format
        if not triplets_parquet.exists() or args.force_download:
            logger.info(f"Creating triplets format for {args.split} split...")
            
            # Use the build_dataset module to create the triplets
            cmd = [
                "python", "-m", "dataset_factory.build_dataset",
                "--preset", preset_path,
                "--split", args.split,
                "--output", str(triplets_parquet)
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        else:
            logger.info(f"Triplets file {triplets_parquet} already exists. Skipping creation.")
    else:
        logger.info("Skipping MS MARCO preparation steps...")
        if not triplets_parquet.exists():
            logger.warning(f"Triplets file {triplets_parquet} not found! You may need to run without --skip_prepare first.")

    # Step 2: Sample the dataset if requested
    if args.samples is not None:
        logger.info(f"Sampling {args.samples} triplets from {triplets_parquet}...")
        
        if not sample_triplets_parquet.exists() or args.force_download:
            # Read the triplets dataset
            if triplets_parquet.exists():
                df = pd.read_parquet(triplets_parquet)
                logger.info(f"Original dataset has {len(df)} triplets")
                
                # Sample the requested number of triplets
                if len(df) > args.samples:
                    random.seed(args.seed)
                    sampled_df = df.sample(n=args.samples, random_state=args.seed)
                    logger.info(f"Sampled {len(sampled_df)} triplets")
                    
                    # Save the sampled dataset
                    sample_triplets_parquet.parent.mkdir(parents=True, exist_ok=True)
                    sampled_df.to_parquet(sample_triplets_parquet, index=False)
                    logger.info(f"Saved sampled dataset to {sample_triplets_parquet}")
                else:
                    logger.warning(f"Requested {args.samples} samples but dataset only has {len(df)} triplets")
                    sample_triplets_parquet = triplets_parquet
            else:
                logger.error(f"Cannot sample from {triplets_parquet} as it does not exist!")
                return
        else:
            logger.info(f"Sampled file {sample_triplets_parquet} already exists. Skipping sampling.")

    # Step 3: Train the model
    if not args.skip_training:
        logger.info("Training the two-tower model...")
        train_cmd = [
            "python", "two_tower_mini.py",
            "--data", str(sample_triplets_parquet),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size)
        ]
        
        if args.wandb:
            train_cmd.append("--wandb")
        
        logger.info(f"Running command: {' '.join(train_cmd)}")
        subprocess.run(train_cmd, check=True)
    else:
        logger.info("Skipping model training step...")
    
    logger.info("All steps completed successfully!")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python
"""
Generate a synthetic dataset and train a two-tower model in one script.

This script demonstrates the full pipeline:
1. Generate synthetic data
2. Convert to parquet
3. Convert to triplets format
4. Train the two-tower model

Usage:
    python generate_and_train.py --size 1000 --epochs 5
"""

import argparse
import subprocess
import os
from pathlib import Path
import logging
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('generate_and_train')

def main():
    parser = argparse.ArgumentParser(description="Generate data and train the two-tower model")
    parser.add_argument('--size', type=int, default=1000, help='Number of positive examples to generate')
    parser.add_argument('--negs', type=int, default=1, help='Number of negatives per positive')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--skip_generate', action='store_true', help='Skip data generation step')
    parser.add_argument('--skip_training', action='store_true', help='Skip model training step')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--config', default='configs/char_tower.yml', help='Base config file to use')
    args = parser.parse_args()

    # Define file paths
    raw_tsv = Path("data/raw/synthetic_pairs.tsv")
    processed_parquet = Path("data/processed/synthetic_pairs.parquet")
    triplets_parquet = Path("data/processed/synthetic_triplets.parquet")

    # Step 1: Generate synthetic dataset
    if not args.skip_generate:
        logger.info(f"Generating synthetic dataset with {args.size} positive examples...")
        gen_cmd = [
            "python", "-m", "dataset_factory.synthetic_dataset_gen",
            "--generate",
            "--n_positive", str(args.size),
            "--neg_per_pos", str(args.negs),
            "--output", "synthetic_pairs.tsv"
        ]
        
        logger.info(f"Running command: {' '.join(gen_cmd)}")
        subprocess.run(gen_cmd, check=True)
        
        # Step 2: Convert to parquet
        logger.info("Converting to parquet format...")
        convert_cmd = [
            "python", "-m", "dataset_factory.synthetic_dataset_gen",
            "--convert",
            "--input", "synthetic_pairs.tsv",
            "--output", "synthetic_pairs.parquet",
            "--format", "pairs"
        ]
        
        logger.info(f"Running command: {' '.join(convert_cmd)}")
        subprocess.run(convert_cmd, check=True)
        
        # Step 3: Convert to triplets format
        logger.info("Converting to triplets format...")
        triplets_cmd = [
            "python", "-m", "dataset_factory.synthetic_dataset_gen",
            "--convert",
            "--input", "data/processed/synthetic_pairs.parquet",
            "--output", "synthetic_triplets.parquet",
            "--format", "triplets"
        ]
        
        logger.info(f"Running command: {' '.join(triplets_cmd)}")
        subprocess.run(triplets_cmd, check=True)
    else:
        logger.info("Skipping data generation steps...")

    # Step 4: Create a temporary config file with synthetic data path
    if not args.skip_training:
        logger.info("Creating temporary config for synthetic data...")
        # Load the base config
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override with synthetic data path and command line arguments
        config['data'] = str(triplets_parquet)
        config['epochs'] = args.epochs
        config['batch_size'] = args.batch_size
        config['use_wandb'] = args.wandb
        
        # Save to a temporary config file
        temp_config_path = "configs/temp_synthetic_config.yml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info("Training the two-tower model...")
        train_cmd = [
            "python", "train.py",
            "--config", temp_config_path
        ]
        
        if args.wandb:
            train_cmd.append("--use_wandb")
        
        logger.info(f"Running command: {' '.join(train_cmd)}")
        try:
            subprocess.run(train_cmd, check=True)
        finally:
            # Clean up temporary config
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
    else:
        logger.info("Skipping model training step...")
    
    logger.info("All steps completed successfully!")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python
"""
Command-line script to train a two-tower model using config files.

Usage Examples:
    # Basic: Run a single experiment with a config file
    python train.py --config configs/char_tower.yml --use_wandb
    
    # Run a single experiment with W&B logging enabled
    python train.py --config configs/char_tower.yml --use_wandb
    
    # Run multiple experiments sequentially
    python train.py --configs configs/char_tower.yml configs/word2vec_skipgram.yml --use_wandb
    
    # Run multiple experiments in parallel (uses all available CPU cores)
    python train.py --configs configs/char_tower.yml configs/word2vec_skipgram.yml --parallel --use_wandb
    
    # Run multiple experiments in parallel with limited workers
    python train.py --configs configs/char_tower.yml configs/word2vec_skipgram.yml --parallel --max-workers 4 --use_wandb
    
    # Run all configs in a directory
    python train.py --config-dir configs/ --use_wandb
    
    # Run all configs in a directory in parallel with custom logging
    python train.py --config-dir configs/ --parallel --log_level DEBUG --log_file custom_log.log --use_wandb
    
    # Run experiments from different config directories
    python train.py --configs configs/embeddings/char_tower.yml configs/encoders/transformer.yml --use_wandb
"""

import argparse
import logging
import sys
import os
import glob
import multiprocessing
from pathlib import Path
from typing import List, Dict, Any
import time

# Import from the twotower package
from twotower import train_model, setup_logging, load_config

def run_experiment(config_path: str, log_level: str, log_file: str, use_wandb: bool) -> None:
    """
    Run a single experiment with the given configuration file.
    
    Args:
        config_path: Path to the configuration file
        log_level: Logging level
        log_file: Path to log file (if None, will be derived from config filename)
        use_wandb: Whether to enable W&B logging
    """
    # Set up logging for this experiment
    if log_file is None:
        config_name = Path(config_path).stem
        log_file = f"two_tower_{config_name}.log"
    
    # Each experiment gets its own logger
    logger = setup_logging(
        log_level=log_level,
        log_file=log_file
    )
    
    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    try:
        config = load_config(config_path)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {str(e)}")
        return
    
    # Override config with command line arguments
    if use_wandb:
        config['use_wandb'] = True
    
    # Print configuration summary
    logger.info("Configuration summary:")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for k, v in value.items():
                logger.info(f"    {k}: {v}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Train model
    try:
        start_time = time.time()
        model = train_model(config)
        end_time = time.time()
        logger.info(f"Training complete! Total time: {end_time - start_time:.2f}s")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a Two-Tower model")
    
    # Config options (mutually exclusive)
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--config", help="Path to a single config YAML file")
    config_group.add_argument("--configs", nargs='+', help="Paths to multiple config YAML files")
    config_group.add_argument("--config-dir", help="Directory containing config YAML files")
    
    # Other options
    parser.add_argument("--log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                      help="Logging level")
    parser.add_argument("--log_file", default=None, help="Path to log file (defaults to config-based name)")
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--parallel", action="store_true", help="Run experiments in parallel")
    parser.add_argument("--max-workers", type=int, default=None, 
                       help="Maximum number of parallel workers (defaults to CPU count)")
    
    args = parser.parse_args()
    
    # Collect all config paths
    config_paths = []
    
    if args.config:
        # Single config file
        config_paths = [args.config]
    elif args.configs:
        # Multiple config files
        config_paths = args.configs
    elif args.config_dir:
        # Directory of config files
        config_dir = Path(args.config_dir)
        if not config_dir.exists() or not config_dir.is_dir():
            print(f"Error: Config directory not found: {args.config_dir}")
            sys.exit(1)
        config_paths = [str(p) for p in config_dir.glob("*.yml")]
        if not config_paths:
            print(f"Error: No YAML files found in directory: {args.config_dir}")
            sys.exit(1)
    
    # Check that all config files exist
    for config_path in config_paths:
        if not Path(config_path).exists():
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
    
    # Set up root logger for the script
    root_logger = setup_logging(
        log_level=args.log_level,
        log_file="train_experiments.log"
    )
    root_logger.info(f"Found {len(config_paths)} experiment configurations")
    for i, path in enumerate(config_paths):
        root_logger.info(f"  {i+1}. {path}")
    
    if args.parallel and len(config_paths) > 1:
        root_logger.info(f"Running {len(config_paths)} experiments in parallel")
        # Set up multiprocessing pool
        max_workers = args.max_workers or multiprocessing.cpu_count()
        # Limit to number of configs
        max_workers = min(max_workers, len(config_paths))
        root_logger.info(f"Using {max_workers} worker processes")
        
        # Create arguments for each experiment
        experiment_args = [
            (config_path, args.log_level, 
             f"two_tower_{Path(config_path).stem}.log" if args.log_file is None else args.log_file, 
             args.use_wandb) 
            for config_path in config_paths
        ]
        
        # Run experiments in parallel
        with multiprocessing.Pool(max_workers) as pool:
            pool.starmap(run_experiment, experiment_args)
    else:
        # Run experiments sequentially
        root_logger.info(f"Running {len(config_paths)} experiments sequentially")
        for config_path in config_paths:
            root_logger.info(f"Starting experiment with config: {config_path}")
            run_experiment(
                config_path=config_path,
                log_level=args.log_level,
                log_file=args.log_file,
                use_wandb=args.use_wandb
            )
    
    root_logger.info("All experiments completed!")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python
"""
Command-line script to train a two-tower model using the config file.

Usage:
    python train.py --config configs/char_tower.yml
"""

import argparse
import logging
import sys
from pathlib import Path

# Import from the twotower package
from twotower import train_model, setup_logging, load_config

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a Two-Tower model")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                      help="Logging level")
    parser.add_argument("--log_file", default="two_tower.log", help="Path to log file")
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
    args = parser.parse_args()
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Set up logging
    logger = setup_logging(
        log_level=args.log_level,
        log_file=args.log_file
    )
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.use_wandb:
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
    model = train_model(config)
    
    logger.info("Training complete!") 
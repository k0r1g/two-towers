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
import json
import datetime
import socket
import yaml
import copy

# Import from the twotower package
from twotower import train_model, setup_logging, load_config

# Try to import torch for hardware info
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def get_hardware_info() -> dict:
    """Collect hardware information for experiment tracking."""
    hardware_info = {
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count()
    }
    
    if HAS_TORCH:
        hardware_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            hardware_info["cuda_device_count"] = torch.cuda.device_count()
            hardware_info["cuda_device_name"] = torch.cuda.get_device_name(0)
            hardware_info["cuda_version"] = torch.version.cuda
    
    return hardware_info

def run_experiment(config_path: str, log_level: str, log_file: str, use_wandb: bool) -> None:
    """
    Run a single experiment with the given configuration file.
    
    Args:
        config_path: Path to the configuration file
        log_level: Logging level
        log_file: Path to log file (if None, will be derived from config filename)
        use_wandb: Whether to enable W&B logging
    """
    # Create a unique experiment ID
    config_name = Path(config_path).stem
    experiment_id = f"tt_{config_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Set up logging for this experiment
    if log_file is None:
        log_file = f"logs/two_tower_{config_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Make sure the log directory exists
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Each experiment gets its own logger
    logger = setup_logging(
        log_level=log_level,
        log_file=log_file
    )
    
    logger.info(f"Starting experiment with ID: {experiment_id}")
    logger.info(f"Experiment parameters:")
    logger.info(f"  Config file: {config_path}")
    logger.info(f"  Log level: {log_level}")
    logger.info(f"  Log file: {log_file}")
    logger.info(f"  Use W&B: {use_wandb}")
    
    # Log hardware information
    hardware_info = get_hardware_info()
    logger.info("Hardware information:")
    for key, value in hardware_info.items():
        logger.info(f"  {key}: {value}")
    
    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    try:
        config = load_config(config_path)
        # Make a copy of the original config for tracking
        original_config = copy.deepcopy(config)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {str(e)}")
        return
    
    # Override config with command line arguments
    if use_wandb:
        config['use_wandb'] = True
    
    # Add experiment metadata to config
    config['experiment'] = {
        'id': experiment_id,
        'timestamp': datetime.datetime.now().isoformat(),
        'config_file': str(config_path),
        'hardware': hardware_info
    }
    
    # Add W&B tags
    if 'wandb' in config:
        if 'tags' not in config['wandb']:
            config['wandb']['tags'] = []
        # Add config file name as a tag
        config['wandb']['tags'].append(f"config_{config_name}")
    else:
        config['wandb'] = {
            "project": "two-tower-retrieval",
            "tags": [f"config_{config_name}"]
        }
    
    # Save the complete config for reference
    config_file = Path(f"logs/config_{experiment_id}.yml")
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Complete configuration saved to {config_file}")
    
    # Print configuration summary
    logger.info("Configuration summary:")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for k, v in value.items():
                logger.info(f"    {k}: {v}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Log modifications from original config
    logger.info("Configuration overrides:")
    for key in config:
        if key in original_config:
            if isinstance(config[key], dict) and isinstance(original_config[key], dict):
                for k in config[key]:
                    if k in original_config[key]:
                        if config[key][k] != original_config[key][k]:
                            logger.info(f"  {key}.{k}: {original_config[key][k]} -> {config[key][k]}")
                    else:
                        logger.info(f"  {key}.{k}: ADDED -> {config[key][k]}")
            elif config[key] != original_config[key]:
                logger.info(f"  {key}: {original_config[key]} -> {config[key]}")
        else:
            logger.info(f"  {key}: ADDED")
    
    # Train model
    try:
        start_time = time.time()
        model = train_model(config)
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"Training complete! Total time: {total_time:.2f}s")
        
        # Create experiment summary
        experiment_summary = {
            "experiment_id": experiment_id,
            "config_file": str(config_path),
            "timestamp": datetime.datetime.now().isoformat(),
            "total_training_time": total_time,
            "success": True,
            "hardware": hardware_info,
        }
        
        # Save experiment summary
        summary_file = Path(f"logs/experiment_summary_{experiment_id}.json")
        with open(summary_file, 'w') as f:
            json.dump(experiment_summary, f, indent=2)
        logger.info(f"Experiment summary saved to {summary_file}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        
        # Create experiment summary for failed run
        experiment_summary = {
            "experiment_id": experiment_id,
            "config_file": str(config_path),
            "timestamp": datetime.datetime.now().isoformat(),
            "total_training_time": time.time() - start_time,
            "success": False,
            "error": str(e),
            "hardware": hardware_info,
        }
        
        # Save experiment summary for failed run
        summary_file = Path(f"logs/experiment_summary_{experiment_id}.json")
        with open(summary_file, 'w') as f:
            json.dump(experiment_summary, f, indent=2)
        logger.info(f"Failed experiment summary saved to {summary_file}")


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
    
    # Log all command-line arguments
    main_logger = logging.getLogger('train_main')
    main_logger.info("Command-line arguments:")
    for arg, value in vars(args).items():
        main_logger.info(f"  {arg}: {value}")
    
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
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    root_logger = setup_logging(
        log_level=args.log_level,
        log_file=f"logs/train_experiments_{timestamp}.log"
    )
    
    # Create an experiment group ID for this run
    experiment_group_id = f"tt_group_{timestamp}"
    
    # Log experiment metadata
    root_logger.info(f"Starting experiment group: {experiment_group_id}")
    root_logger.info(f"Found {len(config_paths)} experiment configurations")
    for i, path in enumerate(config_paths):
        root_logger.info(f"  {i+1}. {path}")
    
    # Log hardware information
    hardware_info = get_hardware_info()
    root_logger.info("Hardware information:")
    for key, value in hardware_info.items():
        root_logger.info(f"  {key}: {value}")
    
    # Save experiment group metadata
    group_metadata = {
        "group_id": experiment_group_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "config_files": config_paths,
        "parallel": args.parallel,
        "max_workers": args.max_workers,
        "hardware": hardware_info
    }
    
    group_metadata_file = Path(f"logs/experiment_group_{experiment_group_id}.json")
    with open(group_metadata_file, 'w') as f:
        json.dump(group_metadata, f, indent=2)
    root_logger.info(f"Experiment group metadata saved to {group_metadata_file}")
    
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
             f"logs/two_tower_{Path(config_path).stem}_{timestamp}.log" if args.log_file is None else args.log_file, 
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
                log_file=f"logs/two_tower_{Path(config_path).stem}_{timestamp}.log" if args.log_file is None else args.log_file,
                use_wandb=args.use_wandb
            )
    
    root_logger.info("All experiments completed!")
    
    # Update group metadata with completion timestamp
    group_metadata["completed_timestamp"] = datetime.datetime.now().isoformat()
    group_metadata["total_runtime"] = (datetime.datetime.now() - 
                                     datetime.datetime.fromisoformat(group_metadata["timestamp"])).total_seconds()
    
    with open(group_metadata_file, 'w') as f:
        json.dump(group_metadata, f, indent=2)
    root_logger.info(f"Updated experiment group metadata with completion information")

if __name__ == "__main__":
    main() 
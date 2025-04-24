#!/usr/bin/env python
"""
Train a two-tower model on the MS MARCO dataset.

This script demonstrates a full training pipeline:
1. Download MS MARCO dataset if not already available
2. Convert to triplets format if needed
3. Sample the dataset to the specified size
4. Train the two-tower model

Usage Examples:
    # Basic: Train on 10,000 samples from MS MARCO with a single preset
    python train_with_msmarco.py --preset presets/multi_pos_multi_neg.yml --samples 10000 --epochs 3 --wandb
    
    # Force redownload of the dataset
    python train_with_msmarco.py --preset presets/multi_pos_multi_neg.yml --force_download --wandb
    
    # Skip data preparation steps (useful for retraining)
    python train_with_msmarco.py --preset presets/multi_pos_multi_neg.yml --skip_prepare --wandb
    
    # Train using dev split instead of train split
    python train_with_msmarco.py --preset presets/multi_pos_multi_neg.yml --split dev --wandb
    
    # Train with multiple presets sequentially
    python train_with_msmarco.py --presets presets/multi_pos_multi_neg.yml presets/single_pos_hard_neg.yml --samples 5000 --wandb
    
    # Train with multiple presets in parallel
    python train_with_msmarco.py --presets presets/multi_pos_multi_neg.yml presets/single_pos_hard_neg.yml --parallel --wandb
    
    # Train with multiple presets in parallel with limited workers
    python train_with_msmarco.py --presets presets/multi_pos_multi_neg.yml presets/single_pos_hard_neg.yml --parallel --max-workers 4 --wandb
    
    # Train with all presets in a directory
    python train_with_msmarco.py --preset-dir presets/ --samples 10000 --wandb
    
    # Train with multiple splits (train and dev) in parallel
    python train_with_msmarco.py --splits train dev --preset presets/multi_pos_multi_neg.yml --parallel --wandb
    
    # Train with multiple splits and multiple presets
    python train_with_msmarco.py --splits train dev --presets presets/multi_pos_multi_neg.yml presets/single_pos_hard_neg.yml --wandb
    
    # Skip the training step (only prepare the data)
    python train_with_msmarco.py --preset presets/multi_pos_multi_neg.yml --skip_training
    
    # Train with Weights & Biases logging enabled
    python train_with_msmarco.py --preset presets/multi_pos_multi_neg.yml --wandb
"""

import argparse
import subprocess
import os
import random
import multiprocessing
import time
from pathlib import Path
import logging
import pandas as pd
import yaml
from typing import List, Dict, Any
import json
import datetime
import wandb

# Import from dataset_factory
from dataset_factory import (
    get_ms_marco_dataset,
    save_dataset_as_parquet,
    transform_and_save_dataset
)

# Import train_model from twotower package for the new pipeline
from twotower import train_model

# Import torch to check for CUDA
import torch

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

def run_experiment(force_download: bool, 
                   skip_prepare: bool, 
                   split: str, 
                   preset_path: str, 
                   samples: int, 
                   seed: int, 
                   epochs: int, 
                   batch_size: int, 
                   skip_training: bool, 
                   use_wandb: bool,
                   config_path: str) -> None:
    """
    Run a single MS MARCO experiment with the given parameters.
    
    Args:
        force_download: Force redownload of MS MARCO dataset
        skip_prepare: Skip data preparation steps
        split: Dataset split to use (train, dev, eval)
        preset_path: Path to preset configuration
        samples: Number of triplets to sample (None = use all)
        seed: Random seed for sampling
        epochs: Number of training epochs
        batch_size: Batch size for training
        skip_training: Skip model training step
        use_wandb: Enable Weights & Biases logging
        config_path: Path to training config YAML file
    """
    # Create a unique experiment ID for tracking
    experiment_id = f"msmarco_{split}_{Path(preset_path).stem}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    experiment_logger = logging.getLogger(f'msmarco_experiment_{Path(preset_path).stem}_{split}')
    experiment_logger.info(f"Starting experiment with ID: {experiment_id}")
    experiment_logger.info(f"Experiment parameters:")
    experiment_logger.info(f"  Split: {split}")
    experiment_logger.info(f"  Preset: {preset_path}")
    experiment_logger.info(f"  Samples: {samples if samples is not None else 'all'}")
    experiment_logger.info(f"  Seed: {seed}")
    experiment_logger.info(f"  Epochs: {epochs}")
    experiment_logger.info(f"  Batch size: {batch_size}")
    experiment_logger.info(f"  Force download: {force_download}")
    experiment_logger.info(f"  Skip prepare: {skip_prepare}")
    experiment_logger.info(f"  Skip training: {skip_training}")
    experiment_logger.info(f"  Use W&B: {use_wandb}")
    experiment_logger.info(f"  Config path: {config_path}")
    
    preset_path = find_preset_file(preset_path)
    
    # Verify the preset file exists and is valid
    if not os.path.exists(preset_path):
        experiment_logger.error(f"Preset file {preset_path} not found. Please specify a valid preset file.")
        return
    
    # Track dataset genealogy
    dataset_genealogy = {
        "experiment_id": experiment_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "ms_marco_split": split,
        "preset_file": str(preset_path),
        "random_seed": seed,
        "sample_size": samples,
        "preprocessing_steps": []
    }
    
    try:
        # Load and validate the preset file
        with open(preset_path, 'r') as f:
            preset_config = yaml.safe_load(f)
            
        # Log preset configuration
        experiment_logger.info(f"Preset configuration from {preset_path}:")
        for key, value in preset_config.items():
            experiment_logger.info(f"  {key}: {value}")
            
        # Add preset details to dataset genealogy
        dataset_genealogy["preset_config"] = preset_config
            
        # Basic validation
        required_keys = ['positive_selector', 'negative_sampler', 'negatives_per_pos']
        missing_keys = [key for key in required_keys if key not in preset_config]
        if missing_keys:
            experiment_logger.warning(f"Preset file {preset_path} is missing required keys: {missing_keys}")
    except Exception as e:
        experiment_logger.warning(f"Error validating preset file {preset_path}: {str(e)}")
        
    # Get the preset name without directory or extension for file naming
    preset_name = Path(preset_path).stem
    
    # Define file paths
    triplets_file = f"{split}_{preset_name}.parquet"
    triplets_parquet = Path(f"data/processed/{triplets_file}")
    
    if samples is not None:
        sample_triplets_file = f"{split}_{preset_name}_sample_{samples}.parquet"
        sample_triplets_parquet = Path(f"data/processed/{sample_triplets_file}")
    else:
        sample_triplets_parquet = triplets_parquet

    # Step 1: Prepare the MS MARCO dataset
    if not skip_prepare:
        experiment_logger.info("Preparing MS MARCO dataset...")
        
        # Step 1.1: Download the dataset if needed
        dataset = get_ms_marco_dataset(force_download=force_download)
        dataset_genealogy["preprocessing_steps"].append({
            "step": "download_dataset",
            "timestamp": datetime.datetime.now().isoformat(),
            "force_download": force_download,
            "dataset_keys": list(dataset.keys()) if dataset else []
        })
        
        # Step 1.2: Save as parquet files
        parquet_files = save_dataset_as_parquet(dataset, force_save=force_download)
        dataset_genealogy["preprocessing_steps"].append({
            "step": "save_as_parquet",
            "timestamp": datetime.datetime.now().isoformat(),
            "force_save": force_download,
            "parquet_files": [str(p) for p in parquet_files] if parquet_files else []
        })
        
        # Step 1.3: Create the triplets format
        if not triplets_parquet.exists() or force_download:
            experiment_logger.info(f"Creating triplets format for {split} split...")
            
            # Use the build_dataset module to create the triplets
            cmd = [
                "python", "-m", "dataset_factory.build_dataset",
                "--preset", preset_path,
                "--split", split,
                "--output", str(triplets_parquet)
            ]
            
            experiment_logger.info(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            dataset_genealogy["preprocessing_steps"].append({
                "step": "create_triplets",
                "timestamp": datetime.datetime.now().isoformat(),
                "command": " ".join(cmd),
                "output_file": str(triplets_parquet),
                "preset_used": str(preset_path)
            })
            
            # Log triplets file info
            if triplets_parquet.exists():
                try:
                    df = pd.read_parquet(triplets_parquet)
                    experiment_logger.info(f"Created triplets file with {len(df)} triplets")
                    experiment_logger.info(f"Triplets file columns: {df.columns.tolist()}")
                    dataset_genealogy["triplets_info"] = {
                        "path": str(triplets_parquet),
                        "row_count": len(df),
                        "columns": df.columns.tolist(),
                        "file_size_bytes": triplets_parquet.stat().st_size
                    }
                except Exception as e:
                    experiment_logger.warning(f"Error reading triplets file: {str(e)}")
        else:
            experiment_logger.info(f"Triplets file {triplets_parquet} already exists. Skipping creation.")
            # Log triplets file info for existing file
            try:
                df = pd.read_parquet(triplets_parquet)
                experiment_logger.info(f"Existing triplets file has {len(df)} triplets")
                experiment_logger.info(f"Triplets file columns: {df.columns.tolist()}")
                dataset_genealogy["triplets_info"] = {
                    "path": str(triplets_parquet),
                    "row_count": len(df),
                    "columns": df.columns.tolist(),
                    "file_size_bytes": triplets_parquet.stat().st_size
                }
            except Exception as e:
                experiment_logger.warning(f"Error reading existing triplets file: {str(e)}")
    else:
        experiment_logger.info("Skipping MS MARCO preparation steps...")
        if not triplets_parquet.exists():
            experiment_logger.warning(f"Triplets file {triplets_parquet} not found! You may need to run without --skip_prepare first.")

    # Step 2: Sample the dataset if requested
    if samples is not None:
        experiment_logger.info(f"Sampling {samples} triplets from {triplets_parquet}...")
        
        if not sample_triplets_parquet.exists() or force_download:
            # Read the triplets dataset
            if triplets_parquet.exists():
                df = pd.read_parquet(triplets_parquet)
                experiment_logger.info(f"Original dataset has {len(df)} triplets")
                
                # Sample the requested number of triplets
                if len(df) > samples:
                    random.seed(seed)
                    sampled_df = df.sample(n=samples, random_state=seed)
                    experiment_logger.info(f"Sampled {len(sampled_df)} triplets")
                    
                    # Save the sampled dataset
                    sample_triplets_parquet.parent.mkdir(parents=True, exist_ok=True)
                    sampled_df.to_parquet(sample_triplets_parquet, index=False)
                    experiment_logger.info(f"Saved sampled dataset to {sample_triplets_parquet}")
                    
                    dataset_genealogy["preprocessing_steps"].append({
                        "step": "sample_dataset",
                        "timestamp": datetime.datetime.now().isoformat(),
                        "input_file": str(triplets_parquet),
                        "output_file": str(sample_triplets_parquet),
                        "input_row_count": len(df),
                        "output_row_count": len(sampled_df),
                        "sample_size": samples,
                        "random_seed": seed
                    })
                    
                    dataset_genealogy["sampled_dataset_info"] = {
                        "path": str(sample_triplets_parquet),
                        "row_count": len(sampled_df),
                        "columns": sampled_df.columns.tolist(),
                        "file_size_bytes": sample_triplets_parquet.stat().st_size
                    }
                else:
                    experiment_logger.warning(f"Requested {samples} samples but dataset only has {len(df)} triplets")
                    sample_triplets_parquet = triplets_parquet
                    
                    dataset_genealogy["preprocessing_steps"].append({
                        "step": "sample_dataset_skipped",
                        "timestamp": datetime.datetime.now().isoformat(),
                        "reason": f"Dataset size {len(df)} smaller than requested sample size {samples}",
                        "using_file": str(triplets_parquet)
                    })
            else:
                experiment_logger.error(f"Cannot sample from {triplets_parquet} as it does not exist!")
                return
        else:
            experiment_logger.info(f"Sampled file {sample_triplets_parquet} already exists. Skipping sampling.")
            # Log info about existing sampled file
            try:
                sampled_df = pd.read_parquet(sample_triplets_parquet)
                experiment_logger.info(f"Existing sampled file has {len(sampled_df)} triplets")
                dataset_genealogy["sampled_dataset_info"] = {
                    "path": str(sample_triplets_parquet),
                    "row_count": len(sampled_df),
                    "columns": sampled_df.columns.tolist(),
                    "file_size_bytes": sample_triplets_parquet.stat().st_size
                }
            except Exception as e:
                experiment_logger.warning(f"Error reading existing sampled file: {str(e)}")

    # Save dataset genealogy to file for reference
    genealogy_file = Path(f"logs/dataset_genealogy_{experiment_id}.json")
    genealogy_file.parent.mkdir(parents=True, exist_ok=True)
    with open(genealogy_file, 'w') as f:
        json.dump(dataset_genealogy, f, indent=2)
    experiment_logger.info(f"Dataset genealogy saved to {genealogy_file}")

    # Step 3: Train the model using the config file
    if not skip_training:
        experiment_logger.info("Training the two-tower model using the config file...")
        
        # Load config file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Log the original config file content
        experiment_logger.info(f"Original configuration from {config_path}:")
        for key, value in config.items():
            if isinstance(value, dict):
                experiment_logger.info(f"  {key}:")
                for k, v in value.items():
                    experiment_logger.info(f"    {k}: {v}")
            else:
                experiment_logger.info(f"  {key}: {value}")
        
        # Override config with command-line arguments and generated data path
        config['data'] = str(sample_triplets_parquet)
        config['epochs'] = epochs
        config['batch_size'] = batch_size
        config['use_wandb'] = use_wandb
        config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Add experiment tracking metadata to config
        config['experiment'] = {
            'id': experiment_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'preset_file': str(preset_path),
            'ms_marco_split': split,
            'sample_size': samples,
            'seed': seed,
            'dataset_genealogy_file': str(genealogy_file)
        }
        
        # Add dataset genealogy directly to config for complete tracking
        config['dataset_genealogy'] = dataset_genealogy
        
        if 'wandb' in config:
            config['wandb']['run_name'] = f"msmarco_{split}_{preset_name}"
            # Add experiment tags to wandb config
            if 'tags' not in config['wandb']:
                config['wandb']['tags'] = []
            config['wandb']['tags'].extend([
                f"split_{split}",
                f"preset_{preset_name}",
                f"samples_{samples if samples is not None else 'all'}"
            ])
        else:
            config['wandb'] = {
                "project": "two-tower-retrieval", 
                "run_name": f"msmarco_{split}_{preset_name}", 
                "entity": None,
                "tags": [
                    f"split_{split}",
                    f"preset_{preset_name}",
                    f"samples_{samples if samples is not None else 'all'}"
                ]
            }
        
        # Log the final config with all overrides
        experiment_logger.info(f"Final training configuration:")
        for key, value in config.items():
            if key not in ['dataset_genealogy']:  # Skip the large genealogy object in logs
                if isinstance(value, dict):
                    experiment_logger.info(f"  {key}:")
                    for k, v in value.items():
                        experiment_logger.info(f"    {k}: {v}")
                else:
                    experiment_logger.info(f"  {key}: {value}")
        
        # Save the complete config for reference
        config_file = Path(f"logs/config_{experiment_id}.yml")
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        experiment_logger.info(f"Complete configuration saved to {config_file}")
        
        start_time = time.time()
        model = train_model(config)
        end_time = time.time()
        experiment_logger.info(f"Training complete! Total time: {end_time - start_time:.2f}s")
        
        # Generate a report if W&B is enabled
        if use_wandb and model is not None and hasattr(wandb, 'run') and wandb.run is not None:
            experiment_logger.info("Generating W&B report for this experiment...")
            try:
                import subprocess
                report_cmd = [
                    "python", "create_report.py",
                    "--run-id", wandb.run.id
                ]
                experiment_logger.info(f"Running command: {' '.join(report_cmd)}")
                report_process = subprocess.run(report_cmd, capture_output=True, text=True)
                
                if report_process.returncode == 0:
                    # Extract the report URL from the output
                    output_lines = report_process.stdout.strip().split('\n')
                    for line in output_lines:
                        if line.startswith("https://"):
                            experiment_logger.info(f"Report generated successfully: {line}")
                            print(f"\n📊 View your experiment report at: {line}\n")
                            break
                else:
                    experiment_logger.warning(f"Failed to generate report: {report_process.stderr}")
            except Exception as e:
                experiment_logger.warning(f"Error generating report: {str(e)}")
    else:
        experiment_logger.info("Skipping model training step...")
    
    experiment_logger.info("Experiment completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Train with MS MARCO dataset")
    # Data preparation options
    parser.add_argument('--force_download', action='store_true', help='Force redownload of MS MARCO dataset')
    parser.add_argument('--skip_prepare', action='store_true', help='Skip data preparation steps')
    
    # Split options (single or multiple)
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument('--split', default='train', choices=['train', 'dev', 'eval'], 
                           help='Dataset split to use')
    split_group.add_argument('--splits', nargs='+', choices=['train', 'dev', 'eval'],
                           help='Multiple dataset splits to use')
    
    # Preset options (single or multiple)
    preset_group = parser.add_mutually_exclusive_group(required=True)
    preset_group.add_argument('--preset', help='Preset configuration for build_dataset')
    preset_group.add_argument('--presets', nargs='+', help='Multiple preset configurations for experiments')
    preset_group.add_argument('--preset-dir', help='Directory containing preset configuration files')
    
    # Sampling options
    parser.add_argument('--samples', type=int, default=None, 
                        help='Number of triplets to sample (default: use all available data)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    
    # Training options
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--skip_training', action='store_true', help='Skip model training step')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--config', default='configs/msmarco_default.yml', help='Path to training config YAML file')
    
    # Parallel execution
    parser.add_argument('--parallel', action='store_true', help='Run experiments in parallel')
    parser.add_argument('--max-workers', type=int, default=None, 
                       help='Maximum number of parallel workers (defaults to CPU count)')
    
    args = parser.parse_args()

    # Log all command-line arguments
    logger.info("Command-line arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    # Collect all preset paths
    preset_paths = []
    if args.preset:
        preset_paths = [args.preset]
    elif args.presets:
        preset_paths = args.presets
    elif args.preset_dir:
        preset_dir = Path(args.preset_dir)
        if not preset_dir.exists() or not preset_dir.is_dir():
            logger.error(f"Preset directory not found: {args.preset_dir}")
            return
        preset_paths = [str(p) for p in preset_dir.glob("*.yml")]
        if not preset_paths:
            logger.error(f"No YAML files found in directory: {args.preset_dir}")
            return
        
    # Collect all splits
    splits = []
    if args.split:
        splits = [args.split]
    elif args.splits:
        splits = args.splits
    else:
        # Default to train split
        splits = ['train']
        
    logger.info(f"Running experiments with:")
    logger.info(f"  Splits: {splits}")
    logger.info(f"  Presets: {[Path(p).name for p in preset_paths]}")
    
    # Create all experiment combinations
    experiments = []
    for split in splits:
        for preset_path in preset_paths:
            experiments.append((split, preset_path))
    
    logger.info(f"Total experiments to run: {len(experiments)}")
    for i, (split, preset_path) in enumerate(experiments):
        logger.info(f"  {i+1}. Split: {split}, Preset: {Path(preset_path).name}")
    
    # Log hardware information
    logger.info("Hardware information:")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"  CUDA device name: {torch.cuda.get_device_name(0)}")
    
    if args.parallel and len(experiments) > 1:
        logger.info("Running experiments in parallel")
        max_workers = args.max_workers or multiprocessing.cpu_count()
        # Limit to number of experiments
        max_workers = min(max_workers, len(experiments))
        logger.info(f"Using {max_workers} worker processes")
        
        # Create arguments for each experiment
        experiment_args = [
            (args.force_download, args.skip_prepare, split, preset_path, 
             args.samples, args.seed, args.epochs, args.batch_size, 
             args.skip_training, args.wandb, args.config)
            for split, preset_path in experiments
        ]
        
        # Run experiments in parallel
        with multiprocessing.Pool(max_workers) as pool:
            pool.starmap(run_experiment, experiment_args)
    else:
        logger.info("Running experiments sequentially")
        for split, preset_path in experiments:
            logger.info(f"Starting experiment with split: {split}, preset: {preset_path}")
            start_time = time.time()
            run_experiment(
                force_download=args.force_download,
                skip_prepare=args.skip_prepare,
                split=split,
                preset_path=preset_path,
                samples=args.samples,
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                skip_training=args.skip_training,
                use_wandb=args.wandb,
                config_path=args.config
            )
            end_time = time.time()
            logger.info(f"Experiment completed in {end_time - start_time:.2f}s")
    
    logger.info("All experiments completed successfully!")

if __name__ == "__main__":
    main() 
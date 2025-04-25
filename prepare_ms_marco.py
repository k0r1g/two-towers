#!/usr/bin/env python
"""
Download and prepare the MS MARCO dataset for two-tower models.

This script:
1. Downloads the MS MARCO dataset
2. Saves the dataset as parquet files
3. Creates various formats needed for training

Usage:
    python prepare_ms_marco.py --force_download
"""

import argparse
import logging
from pathlib import Path
import subprocess

# Import from dataset_factory
from dataset_factory import (
    get_ms_marco_dataset,
    save_dataset_as_parquet,
    transform_and_save_dataset
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('prepare_ms_marco')

def main():
    parser = argparse.ArgumentParser(description="Download and prepare MS MARCO dataset")
    parser.add_argument('--force_download', action='store_true', help='Force redownload of the dataset')
    parser.add_argument('--split', default='train', choices=['train', 'dev', 'eval'], help='Dataset split to prepare')
    parser.add_argument('--skip_triplets', action='store_true', help='Skip creating the triplets format')
    parser.add_argument('--preset', default='presets/multi_pos_multi_neg.yml', help='Preset configuration for build_dataset')
    parser.add_argument('--output', default=None, help='Output file for triplets dataset')
    args = parser.parse_args()

    # Download the dataset
    logger.info("Downloading MS MARCO dataset...")
    dataset = get_ms_marco_dataset(force_download=args.force_download)
    
    # Save as parquet files
    logger.info("Saving dataset splits as parquet files...")
    parquet_files = save_dataset_as_parquet(dataset, force_save=args.force_download)
    
    # Log the created parquet files
    for split, file_path in parquet_files.items():
        logger.info(f"Split {split} saved to {file_path}")
    
    # Create the triplets format if needed
    if not args.skip_triplets:
        logger.info(f"Creating triplets format for {args.split} split...")
        
        # Determine output file name
        if args.output:
            output_file = args.output
        else:
            # Parse the preset name to create a descriptive output name
            preset_name = Path(args.preset).stem
            output_file = f"{args.split}_{preset_name}.parquet"
        
        # Use the build_dataset module to create the triplets
        cmd = [
            "python", "-m", "dataset_factory.build_dataset",
            "--preset", args.preset,
            "--split", args.split,
            "--output", f"data/processed/{output_file}"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        logger.info(f"Triplets dataset created at data/processed/{output_file}")
    
    logger.info("MS MARCO dataset preparation completed!")

if __name__ == "__main__":
    main() 
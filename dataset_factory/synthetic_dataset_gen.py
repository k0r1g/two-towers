#!/usr/bin/env python
"""
Command-line tool for generating synthetic datasets.

Examples:
    # Generate a basic synthetic dataset
    $ python -m dataset_factory.synthetic_dataset_gen --n_positive 1000 --neg_per_pos 2 --output pairs.tsv
    
    # Expand an existing dataset
    $ python -m dataset_factory.synthetic_dataset_gen --expand --input pairs.tsv --output expanded_pairs.tsv --expansion_factor 3
    
    # Generate and convert to parquet in one step
    $ python -m dataset_factory.synthetic_dataset_gen --n_positive 1000 --neg_per_pos 2 --output pairs.tsv --convert_parquet
"""
import argparse
from pathlib import Path
import os

from .synthetic_generators import generate_synthetic_pairs, expand_synthetic_dataset
from .readers import convert_tsv_to_parquet, PROCESSED_DATA_DIR, RAW_DATA_DIR
from .utils import transform_and_save_dataset

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic datasets for two-tower models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main operation type
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--generate', action='store_true', help='Generate a new synthetic dataset')
    group.add_argument('--expand', action='store_true', help='Expand an existing dataset')
    group.add_argument('--convert', action='store_true', help='Convert an existing dataset to a different format')
    
    # Parameters for generation
    parser.add_argument('--n_positive', type=int, default=500, 
                        help='Number of positive pairs to generate (only used with --generate)')
    parser.add_argument('--neg_per_pos', type=int, default=1, 
                        help='Number of negative examples per positive (only used with --generate)')
    
    # Parameters for expansion
    parser.add_argument('--expansion_factor', type=int, default=2,
                        help='How many times larger the expanded dataset should be (only used with --expand)')
    
    # Parameters for conversion
    parser.add_argument('--format', choices=['triplets', 'pairs', 'query_doc_label'], default='triplets',
                        help='Output format for dataset conversion (only used with --convert)')
    
    # Common parameters
    parser.add_argument('--input', type=str, default='pairs.tsv', 
                        help='Input file (for --expand or --convert)')
    parser.add_argument('--output', type=str, default=None, 
                        help='Output file name (default depends on operation)')
    parser.add_argument('--convert_parquet', action='store_true',
                        help='Also convert the result to parquet format')
    
    args = parser.parse_args()
    
    # Set default output if not provided
    if args.output is None:
        if args.generate:
            args.output = "pairs.tsv"
        elif args.expand:
            args.output = "expanded_pairs.tsv"
        elif args.convert:
            args.output = f"{Path(args.input).stem}_{args.format}.parquet"
    
    # Perform the requested operation
    if args.generate:
        print(f"Generating synthetic dataset with {args.n_positive} positive and {args.n_positive * args.neg_per_pos} negative examples...")
        output_path = generate_synthetic_pairs(
            n_positive=args.n_positive,
            n_negative_per_positive=args.neg_per_pos,
            output_file=args.output
        )
        print(f"Synthetic dataset generated and saved to {output_path}")
        
        # Convert to parquet if requested
        if args.convert_parquet:
            parquet_file = f"{Path(args.output).stem}.parquet"
            parquet_path = convert_tsv_to_parquet(args.output, parquet_file)
            print(f"Converted to parquet format: {parquet_path}")
    
    elif args.expand:
        print(f"Expanding dataset {args.input} by factor of {args.expansion_factor}...")
        output_path = expand_synthetic_dataset(
            input_file=args.input,
            output_file=args.output,
            expansion_factor=args.expansion_factor
        )
        print(f"Expanded dataset saved to {output_path}")
        
        # Convert to parquet if requested
        if args.convert_parquet:
            parquet_file = f"{Path(args.output).stem}.parquet"
            parquet_path = convert_tsv_to_parquet(args.output, parquet_file)
            print(f"Converted to parquet format: {parquet_path}")
    
    elif args.convert:
        print(f"Converting dataset {args.input} to {args.format} format...")
        
        # Determine if the input file is an absolute path, exists as is,
        # or needs to be resolved relative to data directories
        input_path = Path(args.input)
        if input_path.is_absolute():
            # Absolute path provided
            input_in_raw = False
        elif os.path.exists(input_path):
            # File exists in current directory
            input_in_raw = False
        elif os.path.exists(PROCESSED_DATA_DIR / input_path):
            # File exists in processed directory
            input_path = PROCESSED_DATA_DIR / input_path
            input_in_raw = False
        else:
            # Assume file is in raw directory
            input_in_raw = True
        
        output_path = transform_and_save_dataset(
            input_file=str(input_path),
            output_file=args.output,
            format_type=args.format,
            input_in_raw=input_in_raw,
            output_in_processed=True
        )
        print(f"Converted dataset saved to {output_path}")

if __name__ == "__main__":
    main() 
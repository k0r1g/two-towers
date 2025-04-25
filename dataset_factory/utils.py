import re
import pandas as pd
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional, Union

from .readers import RAW_DATA_DIR, PROCESSED_DATA_DIR, setup_data_dirs

def flatten_answers(row) -> List[str]:
    return row["answers"] or []   # handles empty lists (HF sets None)

def answer_in_text(text: str, answers: List[str]) -> bool:
    for a in answers:
        # crude but fast: case-insensitive substring
        if a and a.lower() in text.lower():
            return True
    return False

def ngram_set(text: str, n: int = 3) -> Set[str]:
    tokens = text.lower().split()
    return {" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)}

def high_ngram_overlap(p1: str, p2: str, thresh: float = 0.8) -> bool:
    ngrams1, ngrams2 = ngram_set(p1), ngram_set(p2)
    if not ngrams1 or not ngrams2:
        return False
    jacc = len(ngrams1 & ngrams2) / len(ngrams1 | ngrams2)
    return jacc >= thresh

def convert_dataset_format(df: pd.DataFrame, output_format: str = 'triplets') -> pd.DataFrame:
    """
    Convert a dataset from pairs format to triplets format or other formats.
    
    Args:
        df: Input DataFrame with 'query', 'document', 'label' columns
        output_format: Target format ('triplets', 'pairs', 'query_doc_label')
        
    Returns:
        Converted DataFrame
    """
    if output_format == 'triplets':
        # Group by query
        query_groups = df.groupby('query')
        
        triplets = []
        for query, group in query_groups:
            # Get positive and negative documents for this query
            positives = group[group['label'] == 1]['document'].tolist()
            negatives = group[group['label'] == 0]['document'].tolist()
            
            # Skip if we don't have both positives and negatives
            if not positives or not negatives:
                continue
                
            # Create all possible triplets
            for pos in positives:
                for neg in negatives:
                    triplets.append((query, pos, neg))
        
        return pd.DataFrame(triplets, columns=['query', 'positive_doc', 'negative_doc'])
    
    elif output_format == 'query_doc_label':
        # This is the standard format with query, document, label columns
        return df[['query', 'document', 'label']]
    
    elif output_format == 'pairs':
        # This just ensures the columns are in the right order
        return df[['query', 'document', 'label']]
    
    else:
        raise ValueError(f"Unknown output format: {output_format}")

def transform_and_save_dataset(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    format_type: str = 'triplets',
    input_in_raw: bool = True,
    output_in_processed: bool = True
) -> Path:
    """
    Transform a dataset and save it in the specified format.
    
    Args:
        input_file: Input file path (can be relative or absolute)
        output_file: Output file path (can be relative or absolute)
        format_type: Target format ('triplets', 'pairs', 'query_doc_label')
        input_in_raw: If True, interpret input_file as relative to RAW_DATA_DIR
        output_in_processed: If True, interpret output_file as relative to PROCESSED_DATA_DIR
        
    Returns:
        Path to the output file
    """
    # Setup directories
    setup_data_dirs()
    
    # Resolve input path
    if isinstance(input_file, str):
        input_file = Path(input_file)
    
    if input_file.is_absolute():
        input_path = input_file
    elif input_file.exists():
        # File exists in current directory
        input_path = input_file
    elif input_in_raw:
        input_path = RAW_DATA_DIR / input_file
    else:
        input_path = input_file
    
    # Resolve output path
    if isinstance(output_file, str):
        output_file = Path(output_file)
        
    if output_file.is_absolute():
        output_path = output_file
    elif output_in_processed:
        output_path = PROCESSED_DATA_DIR / output_file
    else:
        output_path = output_file
    
    # Read the input file (detect format based on extension)
    print(f"Reading input file from {input_path}")
    if str(input_path).endswith('.parquet'):
        df = pd.read_parquet(input_path)
    elif str(input_path).endswith('.tsv'):
        df = pd.read_csv(input_path, sep='\t', header=None, 
                        names=['query', 'document', 'label'])
    elif str(input_path).endswith('.csv'):
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unknown input file format: {input_path}")
    
    # Transform the dataset
    transformed_df = convert_dataset_format(df, format_type)
    
    # Save the output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save in appropriate format based on extension
    if str(output_path).endswith('.parquet'):
        transformed_df.to_parquet(output_path, index=False)
    elif str(output_path).endswith('.tsv'):
        transformed_df.to_csv(output_path, sep='\t', index=False)
    elif str(output_path).endswith('.csv'):
        transformed_df.to_csv(output_path, index=False)
    else:
        # Default to parquet
        if not str(output_path).endswith('.parquet'):
            output_path = Path(str(output_path) + '.parquet')
        transformed_df.to_parquet(output_path, index=False)
    
    print(f"Transformed dataset from {input_path} to {output_path}")
    print(f"Original shape: {df.shape}, Transformed shape: {transformed_df.shape}")
    
    return output_path

def convert_triplets_to_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a triplets dataset (query, positive_doc, negative_doc) to pairs format (query, document, label).
    
    Args:
        df: DataFrame with triplets format
    
    Returns:
        DataFrame with pairs format
    """
    # Verify that the input DataFrame has the expected columns
    expected_cols = ['query', 'positive_doc', 'negative_doc']
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame is missing required columns: {missing_cols}")
    
    # Create positive pairs
    positive_pairs = df[['query', 'positive_doc']].copy()
    positive_pairs.columns = ['query', 'document']
    positive_pairs['label'] = 1
    
    # Create negative pairs
    negative_pairs = df[['query', 'negative_doc']].copy()
    negative_pairs.columns = ['query', 'document']
    negative_pairs['label'] = 0
    
    # Combine and return
    return pd.concat([positive_pairs, negative_pairs], ignore_index=True) 
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from typing import Dict, Optional, Union

# Define data directories
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_PARQUET_DIR = RAW_DATA_DIR / "parquet"

def setup_data_dirs() -> Path:
    """Create data directories if they don't exist."""
    for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, RAW_PARQUET_DIR]:
        dir_path.mkdir(exist_ok=True, parents=True)
    return RAW_DATA_DIR

def get_ms_marco_dataset(force_download: bool = False) -> Dict:
    """
    Load the Microsoft MS MARCO dataset (v1.1) and save it to the project's data/raw directory.
    
    Args:
        force_download: If True, force redownload even if files exist
        
    Returns:
        The loaded dataset
    """
    # Create data directories
    raw_data_dir = setup_data_dirs()
    cache_dir = str(raw_data_dir)
    
    # Check if dataset already exists
    dataset_files_exist = (raw_data_dir / "microsoft--ms_marco").exists()
    
    if dataset_files_exist and not force_download:
        print(f"Dataset already exists at {raw_data_dir}. Loading from disk...")
    else:
        print(f"Downloading MS MARCO dataset to {raw_data_dir}...")
    
    # The download_mode="force_redownload" parameter can be used to force a redownload
    download_mode = "force_redownload" if force_download else None
    
    # Load the dataset with the specified cache_dir
    dataset = load_dataset(
        "microsoft/ms_marco", 
        "v1.1", 
        cache_dir=cache_dir,
        download_mode=download_mode
    )
    
    print(f"Dataset loaded successfully. Keys: {list(dataset.keys())}")
    print(f"Dataset saved to {raw_data_dir}")
    
    return dataset

def save_dataset_as_parquet(dataset: Dict, force_save: bool = False) -> Dict[str, Path]:
    """
    Save the dataset splits as Parquet files in the raw data directory.
    
    Args:
        dataset: The loaded dataset from get_ms_marco_dataset
        force_save: If True, overwrite existing files
    
    Returns:
        Dictionary mapping split names to file paths
    """
    parquet_dir = RAW_PARQUET_DIR
    parquet_dir.mkdir(exist_ok=True)
    
    parquet_files = {}
    
    for split, data in dataset.items():
        output_file = parquet_dir / f"{split}.parquet"
        parquet_files[split] = output_file
        
        if output_file.exists() and not force_save:
            print(f"Parquet file for {split} already exists at {output_file}")
        else:
            print(f"Saving {split} split to {output_file}...")
            # Save as parquet format
            data.to_parquet(str(output_file))
            print(f"Saved {split} with shape {data.shape} to {output_file}")
    
    return parquet_files

def load_split(split: str = "train") -> pd.DataFrame:
    """Return the MS-MARCO split as a pandas DataFrame."""
    file_path = RAW_PARQUET_DIR / f"{split}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_path} not found. Run your download script first.")
    return pd.read_parquet(file_path)

def load_synthetic_dataset(filename: str = "pairs.parquet") -> pd.DataFrame:
    """
    Load a synthetic dataset from the processed directory.
    
    Args:
        filename: Name of the parquet file to load
        
    Returns:
        DataFrame containing the synthetic dataset
    """
    file_path = PROCESSED_DATA_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_path} not found. Generate your synthetic dataset first.")
    return pd.read_parquet(file_path)

def load_synthetic_tsv(filename: str = "pairs.tsv") -> pd.DataFrame:
    """
    Load a synthetic dataset in TSV format from the raw directory.
    
    Args:
        filename: Name of the TSV file to load
        
    Returns:
        DataFrame containing the synthetic dataset
    """
    file_path = RAW_DATA_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_path} not found. Generate your synthetic dataset first.")
    return pd.read_csv(file_path, sep='\t', header=None, 
                      names=['query', 'document', 'label'])

def convert_tsv_to_parquet(
    input_file: Union[str, Path] = "pairs.tsv", 
    output_file: Union[str, Path] = "pairs.parquet"
) -> Path:
    """
    Convert a TSV file to Parquet format.
    
    Args:
        input_file: Path to the input TSV file (relative to RAW_DATA_DIR)
        output_file: Path to the output Parquet file (relative to PROCESSED_DATA_DIR)
        
    Returns:
        Path to the output Parquet file
    """
    # Setup directories
    setup_data_dirs()
    
    # Resolve paths
    if isinstance(input_file, str):
        input_path = RAW_DATA_DIR / input_file
    else:
        input_path = input_file
        
    if isinstance(output_file, str):
        output_path = PROCESSED_DATA_DIR / output_file
    else:
        output_path = output_file
    
    # Read the TSV file
    df = pd.read_csv(input_path, sep='\t', header=None, 
                     names=['query', 'document', 'label'])
    
    # Convert label to integer if needed
    df['label'] = df['label'].astype(int)
    
    # Save as parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    print(f"Converted {input_path} to {output_path}")
    return output_path 
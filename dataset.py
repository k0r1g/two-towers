import os
from datasets import load_dataset
from pathlib import Path

# Define data directories
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

def setup_data_dirs():
    """Create data directories if they don't exist."""
    for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
        dir_path.mkdir(exist_ok=True, parents=True)
    return RAW_DATA_DIR

def get_dataset(force_download=False):
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

def save_dataset_as_parquet(dataset, force_save=False):
    """
    Save the dataset splits as Parquet files in the raw data directory.
    
    Args:
        dataset: The loaded dataset from get_dataset
        force_save: If True, overwrite existing files
    
    Returns:
        Dictionary mapping split names to file paths
    """
    parquet_dir = RAW_DATA_DIR / "parquet"
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

if __name__ == "__main__":
    # Example usage
    ds = get_dataset()
    
    # Save as Parquet files
    parquet_files = save_dataset_as_parquet(ds)
    
    # Print information about the dataset
    print("\nDataset structure:")
    for split, data in ds.items():
        print(f"{split}: {data}")
        if hasattr(data, "features") and data.features:
            print(f"Features: {data.features}")
        if hasattr(data, "shape") and data.shape:
            print(f"Shape: {data.shape}")
        print()

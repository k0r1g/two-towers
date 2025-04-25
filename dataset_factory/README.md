# Dataset Factory

This module provides utilities for working with datasets for two-tower retrieval models.

## Directory Structure

The dataset factory is organized to work with the following directory structure:

```
data/
├── raw/             # Raw datasets (MS MARCO, synthetic TSV files)
│   └── parquet/     # Raw datasets in parquet format
└── processed/       # Processed datasets ready for training
```

## Features

- Load and process the MS MARCO dataset
- Generate synthetic datasets for quick prototyping
- Convert between different dataset formats (pairs, triplets)
- Handle file format conversions (TSV to parquet)

## Main Components

### Real-world Datasets

For working with the MS MARCO dataset:

```python
from dataset_factory import get_ms_marco_dataset, save_dataset_as_parquet, load_split

# Download and load the MS MARCO dataset
dataset = get_ms_marco_dataset()

# Save as parquet files
save_dataset_as_parquet(dataset)

# Load a specific split
train_df = load_split("train")
```

### Synthetic Dataset Generation

For generating synthetic datasets:

```python
from dataset_factory import generate_synthetic_pairs, expand_synthetic_dataset

# Generate a synthetic dataset
output_path = generate_synthetic_pairs(
    n_positive=1000,
    n_negative_per_positive=2,
    output_file="pairs.tsv"
)

# Expand an existing dataset
expanded_path = expand_synthetic_dataset(
    input_file="pairs.tsv",
    output_file="expanded_pairs.tsv",
    expansion_factor=3
)
```

### Format Conversion

```python
from dataset_factory import convert_tsv_to_parquet, transform_and_save_dataset

# Convert TSV to parquet
parquet_path = convert_tsv_to_parquet("pairs.tsv", "pairs.parquet")

# Transform to triplets format
triplets_path = transform_and_save_dataset(
    input_file="pairs.tsv",
    output_file="triplets.parquet",
    format_type="triplets"
)
```

## Command-line Usage

The module provides command-line tools for dataset operations:

### Build a Dataset

```bash
python -m dataset_factory.build_dataset \
    --preset presets/multi_pos_multi_neg.yml \
    --split train \
    --output data/processed/multi_pos_multi_neg.parquet
```

### Generate Synthetic Dataset

```bash
# Generate a new synthetic dataset
python -m dataset_factory.synthetic_dataset_gen --generate \
    --n_positive 1000 --neg_per_pos 2 --output pairs.tsv

# Expand an existing dataset
python -m dataset_factory.synthetic_dataset_gen --expand \
    --input pairs.tsv --output expanded_pairs.tsv --expansion_factor 3

# Convert a dataset format
python -m dataset_factory.synthetic_dataset_gen --convert \
    --input pairs.tsv --output triplets.parquet --format triplets
``` 
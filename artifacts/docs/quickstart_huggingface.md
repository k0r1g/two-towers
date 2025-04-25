# HuggingFace Hub Quick Start Guide

This guide provides step-by-step instructions for using the Two-Tower model with HuggingFace Hub integration.

## Prerequisites

1. Install the required packages:

```bash
pip install huggingface_hub
```

2. Login to HuggingFace Hub:

```bash
huggingface-cli login
```

## Creating Repositories

### Option 1: Command-line Setup

Use the provided CLI tool to set up all necessary repositories:

```bash
# Create model repository only
python -m tools.hf_setup --repo-name mlx7-two-tower

# Create model, dataset, and demo repositories
python -m tools.hf_setup --repo-name mlx7-two-tower --create-all

# Create private repositories
python -m tools.hf_setup --repo-name mlx7-two-tower --create-all --private
```

### Option 2: Web Interface

1. Go to [HuggingFace Hub New Model](https://huggingface.co/new)
2. Select the owner (you or your organization)
3. Enter "mlx7-two-tower" as the model name
4. Choose public or private
5. Select a license
6. Click "Create Model"

## Training and Uploading a Model

### Option 1: Integrated Training and Upload

Use the `--push_to_hub` flag during training to automatically upload the model to HuggingFace Hub:

```bash
python -m twotower.train --config configs/char_tower.yml --push_to_hub --hub_repo_id "username/mlx7-two-tower"
```

### Option 2: Configuration-Based Upload

Create a configuration file with HuggingFace Hub settings:

```yaml
# configs/char_tower_hf.yml
extends: configs/char_tower.yml

huggingface:
  push_to_hub: true
  repo_id: mlx7-two-tower
  private: false
```

Then train with this configuration:

```bash
python -m twotower.train --config configs/char_tower_hf.yml
```

### Option 3: Uploading After Training

If you have an existing trained model, you can upload it manually:

```bash
python -m tools.huggingface upload-model username/mlx7-two-tower checkpoints/best_model.pt
```

## Loading Models from HuggingFace Hub

### Python API

```python
from twotower import load_model_from_hub
from twotower.encoders import TwoTower
from twotower.tokenisers import CharTokeniser

# Load a model from HuggingFace Hub
model, tokenizer, config = load_model_from_hub(
    repo_id="username/mlx7-two-tower",
    model_class=TwoTower,
    tokenizer_class=CharTokeniser
)

# Use the model for inference
# ...
```

### Specific Model Version

You can load a specific version of a model by specifying the revision parameter:

```python
model, tokenizer, config = load_model_from_hub(
    repo_id="username/mlx7-two-tower",
    model_class=TwoTower,
    tokenizer_class=CharTokeniser,
    revision="v1.0.0"  # Branch name, tag, or commit hash
)
```

## Uploading and Downloading Datasets

### Uploading a Dataset

```bash
python -m tools.huggingface upload-dataset username/mlx7-two-tower-data data/processed/classic_triplets.parquet
```

### Downloading a Dataset

```python
from twotower import download_dataset_from_hub

# Download a specific file
dataset_path = download_dataset_from_hub(
    repo_id="username/mlx7-two-tower-data",
    output_dir="data/downloaded",
    filename="classic_triplets.parquet"
)
```

## Next Steps

For more detailed information, see the complete [HuggingFace Hub Integration Documentation](huggingface.md). 
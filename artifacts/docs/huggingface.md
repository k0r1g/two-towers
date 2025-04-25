# HuggingFace Hub Integration

The Two-Tower model can be seamlessly integrated with the HuggingFace Hub, allowing you to:

- Create model, dataset, and demo repositories
- Upload models, datasets, and configurations
- Download and load models and datasets
- Share your trained models with the community

## Setup

Before using the HuggingFace Hub integration, you need to authenticate with your HuggingFace account. You have two options:

### Option 1: Login with the CLI (Recommended)
```bash
huggingface-cli login
```

This will prompt you for your HuggingFace authentication token, which you can obtain from your [HuggingFace account settings](https://huggingface.co/settings/tokens).

### Option 2: Use .env file
Add your HuggingFace token to the `.env` file in the project root:

```
# Hugging Face
HUGGINGFACE_ACCESS_TOKEN=hf_your_token_here
```

Scripts that use `load_dotenv()` will automatically load this token. Make sure the `.env` file exists and contains the correct token.

## Creating Repositories

### Command-line Setup

The easiest way to set up HuggingFace repositories for your project is to use the provided CLI tool:

```bash
python -m tools.hf_setup --repo-name mlx7-two-tower --create-all
```

This will create:
- A model repository at `username/mlx7-two-tower`
- A dataset repository at `username/mlx7-two-tower-data`
- A demo space at `username/mlx7-two-tower-demo`

Options:
- `--repo-name`: Base name for the repositories (default: `mlx7-two-tower`)
- `--private`: Make the repositories private
- `--create-all`: Create model, dataset, and space repositories (if omitted, only creates model repository)

### Programmatic Setup

You can also create repositories programmatically:

```python
from tools.huggingface import setup_repository

# Create a model repository
model_repo_url = setup_repository(
    repo_id="username/mlx7-two-tower",
    private=False,
    repo_type="model"
)

# Create a dataset repository
dataset_repo_url = setup_repository(
    repo_id="username/mlx7-two-tower-data",
    private=False,
    repo_type="dataset"
)
```

## Uploading Datasets

### Using the Dataset Uploader Script

The simplest way to upload curated datasets to HuggingFace is using the provided script:

```bash
python tools/create_dataset_repo.py --repo-name mlx7-two-tower-data --data-dir data/processed/curated
```

This script will:
1. Create the dataset repository if it doesn't exist
2. Generate a comprehensive dataset card (README.md) with metadata
3. Upload all parquet files from the specified directory
4. Organize the datasets in a `data/` folder in the repository

Options:
- `--repo-name`: Repository name (default: `mlx7-two-tower-data`)
- `--private`: Make the repository private
- `--data-dir`: Directory containing the curated datasets (default: `data/processed/curated`)
- `--skip-upload`: Skip uploading the datasets (just create the repository and card)

#### Authentication for the Dataset Script
The script uses the `HUGGINGFACE_ACCESS_TOKEN` environment variable from your `.env` file. If you encounter authentication errors:

1. Verify your `.env` file contains the correct token:
   ```
   HUGGINGFACE_ACCESS_TOKEN=hf_your_token_here
   ```

2. If still facing issues, try using the CLI authentication instead:
   ```
   huggingface-cli login
   ```
   
3. For debugging, you can verify your token is being loaded with:
   ```python
   python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('Token loaded:', os.environ.get('HUGGINGFACE_ACCESS_TOKEN') is not None)"
   ```

### Manual Dataset Uploads

You can also upload individual dataset files using the command-line utility:

```bash
# Upload a dataset
python -m tools.huggingface upload-dataset username/mlx7-two-tower-data data/processed/curated/classic_triplets.parquet
```

## Uploading Models

### During Training

The easiest way to upload a model is to enable the `push_to_hub` flag during training:

```bash
python -m twotower.train --config configs/char_tower.yml --push_to_hub --hub_repo_id "username/mlx7-two-tower"
```

Options:
- `--push_to_hub`: Enable pushing to HuggingFace Hub
- `--hub_repo_id`: Repository ID (default: `mlx7-two-tower`)
- `--hub_private`: Make the repository private

### Via Configuration

You can also configure the HuggingFace Hub integration in your YAML configuration file:

```yaml
# configs/char_tower_with_hf.yml
extends: configs/char_tower.yml

huggingface:
  push_to_hub: true
  repo_id: username/mlx7-two-tower
  private: false
```

### Manual Upload

If you want to upload a model after training, you can use the provided utility functions:

```python
from twotower.huggingface import save_and_upload

# Upload a model to HuggingFace Hub
repo_url = save_and_upload(
    model=model,
    tokenizer=tokenizer,
    config=config,
    repo_id="username/mlx7-two-tower",
    local_dir="hub_export",
    private=False
)
```

### Command-line Upload

You can also use the command-line utilities to upload files:

```bash
# Upload a model file
python -m tools.huggingface upload-model username/mlx7-two-tower checkpoints/best_model.pt

# Upload a configuration
python -m tools.huggingface upload-config username/mlx7-two-tower configs/char_tower.yml
```

## Loading Models from Hub

To load a trained model from the HuggingFace Hub:

```python
from twotower.huggingface import load_model_from_hub
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

## Downloading Datasets

To download a dataset from the HuggingFace Hub:

```python
from twotower.huggingface import download_dataset_from_hub

# Download a specific file
dataset_path = download_dataset_from_hub(
    repo_id="username/mlx7-two-tower-data",
    output_dir="data/downloaded",
    filename="classic_triplets.parquet"
)

# Download an entire repository
dataset_dir = download_dataset_from_hub(
    repo_id="username/mlx7-two-tower-data",
    output_dir="data/downloaded"
)
```

## Additional Features

### Versioning

HuggingFace Hub supports versioning through Git branches and tags. You can use the `revision` parameter to specify a specific version:

```python
# Load a specific model version
model, tokenizer, config = load_model_from_hub(
    repo_id="username/mlx7-two-tower",
    model_class=TwoTower,
    tokenizer_class=CharTokeniser,
    revision="v1.0.0"  # Branch name, tag, or commit hash
)
```

### Cloning Repositories

If you need full Git access to a repository:

```python
from tools.huggingface import clone_repository

# Clone a repository
repo = clone_repository(
    repo_id="username/mlx7-two-tower",
    local_dir="local_repo",
    repo_type="model"
)
``` 
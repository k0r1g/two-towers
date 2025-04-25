# Training Commands with W&B and HuggingFace Integration

This document provides examples of commands for training the Two-Tower model with Weights & Biases logging and HuggingFace Hub integration. Each command uses a small dataset to allow for quick testing and verification.

## Basic Training Commands (`train.py`)

The main training script supports logging to W&B and uploading models to HuggingFace Hub:

```bash
# Train with a small dataset using a pre-defined config
python train.py --config configs/test_small.yml --use_wandb

# Use a custom configuration but enable W&B
python train.py --config configs/char_tower.yml --use_wandb

# Enable HuggingFace Hub upload
python train.py --config configs/char_tower.yml --use_wandb --push_to_hub --hub_repo_id "username/model-name"

# Run multiple experiments sequentially
python train.py --configs configs/char_tower.yml configs/word2vec_skipgram.yml --use_wandb

# Run multiple experiments in parallel
python train.py --configs configs/char_tower.yml configs/word2vec_skipgram.yml --parallel --use_wandb
```

## MS MARCO Training Commands (`train_with_msmarco.py`)

For training with the MS MARCO dataset, use these commands:

```bash
# Train on a tiny sample from MS MARCO with W&B logging
python train_with_msmarco.py --preset presets/classic.yml --samples 10 --epochs 1 --wandb

# Skip data preparation if you've already run it once
python train_with_msmarco.py --preset presets/classic.yml --samples 10 --epochs 1 --skip_prepare --wandb

# Use different presets
python train_with_msmarco.py --preset presets/multi_pos_multi_neg.yml --samples 10 --wandb

# Train on different splits
python train_with_msmarco.py --preset presets/classic.yml --split dev --samples 10 --wandb

# Run multiple presets in parallel
python train_with_msmarco.py --presets presets/classic.yml presets/multi_pos_multi_neg.yml --samples 10 --parallel --wandb
```

## Creating Minimal Configuration Files

For testing, you can create minimal configuration files:

```yaml
# configs/test_small.yml
# Configuration for minimal testing with HuggingFace Hub integration and W&B
extends: configs/char_tower.yml

# Dataset path (extremely small dataset for testing)
data: data/processed/train_multi_pos_multi_neg_sample_10.parquet

# Training parameters (minimal for quick testing)
epochs: 1
batch_size: 32
device: cpu  # Use CPU for testing to avoid CUDA errors
learning_rate: 0.001  # Important: ensure this is a float value

# HuggingFace Hub settings
huggingface:
  push_to_hub: true
  repo_id: test-two-tower-temp
  private: true

# Weights & Biases settings
use_wandb: true
wandb:
  run_name: test_small_dataset
  project: two-tower-retrieval
  tags: ["test", "minimal"]
```

## Troubleshooting

If you encounter errors with `train_with_msmarco.py` related to learning rate conversion, ensure that in your configuration file, the learning rate is specified as a float value (e.g., `0.001`) rather than a string.

## Viewing Results

After training, you can:

1. View your W&B dashboard to see training metrics
2. Access your model on HuggingFace Hub at the specified repository URL 
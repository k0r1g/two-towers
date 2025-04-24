# Two-Tower Configuration Reference

This document provides a comprehensive reference for all configuration options available in the Two-Tower retrieval system.

## Configuration System

The Two-Tower system uses a hierarchical YAML-based configuration system with the following features:

- **Inheritance**: Configs can extend other configs using the `extends` property
- **Environment Variables**: Override any setting with environment variables using the `TWOTOWER_` prefix
- **Command-line Overrides**: Many scripts allow overriding config values with command-line arguments

## Basic Usage

```bash
# Using a specific config file
python train.py --config configs/char_tower.yml

# Overriding values via command line
python train.py --config configs/char_tower.yml --epochs 10 --batch_size 512

# Overriding values via environment variables
TWOTOWER_BATCH_SIZE=512 TWOTOWER_EPOCHS=10 python train.py --config configs/char_tower.yml
```

## Inheritance Example

Config files can inherit from other configs using the `extends` property:

```yaml
# In configs/my_custom_config.yml
extends: default_config.yml

# Override specific settings
embedding:
  embedding_dim: 128
```

## Environment Variable Overrides

Any configuration setting can be overridden with environment variables using the `TWOTOWER_` prefix:

- Simple values: `TWOTOWER_BATCH_SIZE=512`
- Nested values: `TWOTOWER_WANDB__PROJECT=my-project`

## Core Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `data` | string | data/processed/classic_triplets.parquet | Path to the training data file (parquet format) |
| `device` | string | "cuda" | Device to run training on (falls back to "cpu" if CUDA not available) |
| `checkpoint_dir` | string | "checkpoints" | Directory to save model checkpoints |
| `batch_size` | integer | 256 | Training batch size |
| `epochs` | integer | 3 | Number of training epochs |
| `learning_rate` | float | 1e-3 | Learning rate for optimizer |
| `max_sequence_length` | integer | 64 | Maximum sequence length for inputs |
| `use_wandb` | boolean | false | Enable Weights & Biases logging |

## Model Architecture Options

### Tokeniser Configuration

```yaml
tokeniser:
  type: "char"       # Options: "char", "word", "wordpiece", "bpe"
  max_len: 64        # Maximum sequence length
```

### Embedding Configuration

```yaml
embedding:
  type: "lookup"     # Options: "lookup", "pretrained", "positional"
  embedding_dim: 64  # Embedding dimension
```

### Encoder Configuration

```yaml
encoder:
  arch: "mean"       # Options: "mean", "cnn", "rnn", "transformer"
  hidden_dim: 128    # Hidden dimension size
  tied_weights: true # Whether to share weights between query and doc towers
```

### Loss Function Configuration

```yaml
loss:
  type: "triplet"    # Options: "triplet", "contrastive", "cosine"
  margin: 0.2        # Margin for triplet/contrastive loss
```

### Optimizer Configuration

```yaml
optimizer:
  type: "adamw"      # Options: "adamw", "adam", "sgd"
  lr: 1e-3           # Learning rate
```

### Weights & Biases Configuration

```yaml
wandb:
  project: "two-tower-retrieval"  # W&B project name
  entity: "username"              # W&B username or team name
  tags: ["experiment", "v1"]      # Tags for the run
```

## Complete Example Configuration

```yaml
# Complete example configuration
extends: default_config.yml

# Data and resources (override defaults if needed)
data: data/processed/my_triplets.parquet
device: cuda
checkpoint_dir: checkpoints/my_experiment

# Tokeniser
tokeniser:
  type: char
  max_len: 64

# Embedding
embedding:
  type: lookup
  embedding_dim: 128

# Encoder
encoder:
  arch: cnn
  hidden_dim: 256
  tied_weights: true

# Loss
loss:
  type: triplet
  margin: 0.2

# Training
batch_size: 512
epochs: 10
optimizer:
  type: adamw
  lr: 0.0001

# W&B
use_wandb: true
wandb:
  project: my-retrieval-project
  entity: my-username
  tags: ["production", "cnn-architecture"]
```

## Advanced Configuration

### Dataset Configuration

```yaml
dataset:
  load_to_memory: true  # Whether to load the entire dataset into memory
```

### Experiment Tracking

```yaml
experiment:
  id: "experiment_v1"  # Custom experiment ID
  tags: ["baseline"]   # Tags for tracking
``` 
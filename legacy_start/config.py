"""
Configuration settings for two-tower experiments.
Contains project-wide constants and default values.

DEPRECATED: This file is deprecated in favor of YAML configuration files in the configs/ directory.
The default configuration is now in configs/default_config.yml.
This file is kept for backward compatibility and will be removed in the future.
"""

# W&B settings
WANDB_PROJECT = "two-tower-retrieval"
WANDB_ENTITY = "azuremis"  # Set to your username or team name if you have a default

# Model defaults
DEFAULT_EMBEDDING_DIM = 64
DEFAULT_HIDDEN_DIM = 128
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_EPOCHS = 3

# Other project constants
CHECKPOINTS_DIR = "checkpoints"
MAX_SEQUENCE_LENGTH = 64 
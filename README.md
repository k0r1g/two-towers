# Two Tower Retrieval Model

This repository contains a minimal implementation of a Two-Tower (Dual Encoder) neural network for document retrieval with comprehensive Weights & Biases logging and automatic report generation.

## Features

- Character-level vocabulary (easy to inspect)
- Custom contrastive triplet loss for better retrieval performance
- Automatic W&B logging and visualization
- Automatic report generation after each run
- Support for both TSV and Parquet data formats
- Centralized configuration management

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/two-towers.git
cd two-towers
```

2. Install the required packages:
```bash
pip install torch pandas tqdm wandb python-dotenv wandb-workspaces
```

3. Create a `.env` file with your W&B API key (optional):
```
WANDB_API_KEY=your_api_key_here
```

## Configuration

The project uses a centralized configuration system with three levels of precedence:

1. **Default values** in `config.py` (lowest precedence)
2. **Environment variables** in `.env` file (middle precedence)
3. **Command-line arguments** (highest precedence)

### Customizing Configuration

Edit the `config.py` file to set default values for your project:

```python
# W&B settings
WANDB_PROJECT = "two-tower-retrieval"
WANDB_ENTITY = "your-username"  # Set to your username or team name

# Model defaults
DEFAULT_EMBEDDING_DIM = 64
DEFAULT_HIDDEN_DIM = 128
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_EPOCHS = 3

# Other project constants
CHECKPOINTS_DIR = "checkpoints"
MAX_SEQUENCE_LENGTH = 64
```

## Data Format

The model expects training data in one of two formats:

1. **TSV format** with three columns:
```
query<TAB>document<TAB>label
```
Where *label* is 1 for a relevant (positive) pair and 0 for a random (negative) pair.

2. **Parquet format** with the same three columns: `query`, `document`, and `label`.

## Running the Model

Run the model with:

```bash
python two_tower_mini.py --data pairs.parquet --epochs 3 --wandb
```

### Command-line options:

- `--data`: Path to input data file (default: 'pairs.parquet')
- `--epochs`: Number of training epochs (default: from config.py)
- `--batch_size`: Batch size for training (default: from config.py)
- `--learning_rate`: Learning rate (default: from config.py)
- `--device`: Device to use ('cpu' or 'cuda:0', default: 'cpu')
- `--wandb`: Enable Weights & Biases logging
- `--wandb_project`: W&B project name (default: from .env or config.py)
- `--wandb_entity`: W&B entity (username or team, default: from .env or config.py)
- `--wandb_run_name`: Custom name for the W&B run
- `--log_level`: Logging level (DEBUG, INFO, WARNING, ERROR, default: INFO)

## Automatic W&B Reports

When running with `--wandb`, the script automatically generates a comprehensive Weights & Biases report after training. The report includes:

- Training loss curves
- Similarity metrics visualizations
- Performance metrics (training speed, timing breakdowns)
- Gradient norms
- Retrieval results for your search query
- Data examples

You can view the report in your W&B dashboard under the "Reports" tab, or by following the link printed in the terminal after training completes.

## Manual Report Generation

You can also generate reports manually with:

```bash
python create_report.py --title "My Custom Report" --description "Analysis of experiment results"
```

Options:
- `--project`: W&B project name (default: from config.py)
- `--entity`: W&B username or team name (default: from config.py)
- `--title`: Custom title for the report (optional)
- `--description`: Description for the report (optional)
- `--run-id`: W&B run ID to focus on in the report (optional)

## Architecture

The Two-Tower model consists of:
1. **Query Tower**: Processes query text into embedding vectors
2. **Document Tower**: Processes document text into embedding vectors

Both towers have identical architectures but maintain separate weights. The model is trained to maximize the similarity between queries and relevant documents while minimizing similarity with non-relevant documents.

## License

[MIT License](LICENSE)






Plan out pipeline: 




Plan out data model: 



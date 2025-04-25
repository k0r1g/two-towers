# Two-Tower Retrieval Model

This repository contains a modular implementation of a Two-Tower (Dual Encoder) neural network for document retrieval.

## Features

- **Modular Design**: Each component (tokenization, embedding, encoding) is implemented as a separate module
- **Config-Driven**: All model and training parameters defined in YAML configuration files
- **Easily Extensible**: Adding new tokenizers, embeddings, or encoders only requires implementing a new class
- **Standard IR Metrics**: Comprehensive evaluation metrics (Precision@K, Recall@K, MRR, NDCG)
- **Unified Search Interface**: Common interface for different search implementations
- **CLI Tools**: Command-line tools for building indices and retrieving documents

## Directory Structure

```
two-towers/
│
├─ twotower/                   # Core model training code
│   ├─ tokenisers.py           # Tokenization (Stage 1)
│   ├─ embeddings.py           # Embedding layers (Stage 2)
│   ├─ encoders.py             # Encoder towers (Stage 3)
│   ├─ losses.py               # Loss functions (Stage 4)
│   ├─ dataset.py              # Dataset handling
│   ├─ train.py                # Training orchestration
│   ├─ utils.py                # Utilities and helpers
│   └─ evaluate.py             # Evaluation metrics
│
├─ inference/                  # Inference and retrieval code
│   ├─ search/                 # Search implementations
│   │   ├─ base.py             # Base search interface
│   │   ├─ glove.py            # GloVe-based search
│   │   └─ two_tower.py        # Two-Tower model search
│   ├─ cli/                    # Command-line tools
│   │   └─ retrieve.py         # Document retrieval CLI
│   └─ examples/               # Example scripts
│
├─ configs/                    # Configuration files
│   ├─ default_config.yml      # Base configuration
│   ├─ char_tower.yml          # Character-level model config
│   └─ word2vec_skipgram.yml   # Word2Vec embedding config
│
├─ docs/                       # Documentation
├─ tools/                      # Utility scripts
└─ artifacts/                  # Project artifacts and documentation
```

## Getting Started

### Installation

Install the package in development mode:

```bash
pip install -e .
```

### Training a Model

To train a model using a configuration file:

```bash
python train.py --config configs/char_tower.yml
```

To enable Weights & Biases logging:

```bash
python train.py --config configs/char_tower.yml --use_wandb
```

### Retrieving Documents

First, build an index of document vectors:

```bash
python -m inference.cli.retrieve build-index --model checkpoints/best_model.pt --documents my_documents.txt --output document_index.pkl
```

Then, retrieve documents for a query:

```bash
python -m inference.cli.retrieve search --model checkpoints/best_model.pt --index document_index.pkl --query "your search query"
```

### Using Search in Python

```python
from inference.search import GloVeSearch, TwoTowerSearch

# GloVe-based search
glove_search = GloVeSearch(model_name='glove-wiki-gigaword-50')
glove_search.index_documents(documents)
results = glove_search.search("query text", top_k=5)

# Two-Tower search
from twotower import load_checkpoint
checkpoint = load_checkpoint("model.pt")
model = checkpoint["model"]
tokenizer = checkpoint["tokenizer"]
two_tower_search = TwoTowerSearch(model, tokenizer)
two_tower_search.index_documents(documents)
results = two_tower_search.search("query text", top_k=5)
```

### Evaluating Models

```python
from twotower.evaluate import evaluate_model, print_evaluation_results

results = evaluate_model(
    model=model,
    test_data=test_data,
    tokenizer=tokenizer,
    metrics=['precision', 'recall', 'mrr', 'ndcg'],
    k_values=[1, 3, 5, 10]
)

print_evaluation_results(results)
```

## Configuration System

The Two-Tower system uses a hierarchical YAML-based configuration system with:

- **Inheritance**: Configs can extend other configs using the `extends` property
- **Environment Variables**: Override settings with `TWOTOWER_` prefixed environment variables
- **Command-line Overrides**: Override configs with command-line arguments

For a complete configuration reference, see [artifacts/docs/config.md](artifacts/docs/config.md).

## Architecture

The code follows a 5-stage pipeline:

1. **Tokenization**: Converts text to token IDs (`tokenisers.py`)
2. **Embedding**: Maps token IDs to dense vectors (`embeddings.py`)
3. **Encoding**: Transforms token embeddings into a single vector (`encoders.py`)
4. **Loss Function**: Defines the training objective (`losses.py`)
5. **Training**: Orchestrates the training process (`train.py`)

## Extending the Model

The modular design makes it easy to extend the model with new components:

### Adding a New Tokenizer

1. Create a new class that inherits from `BaseTokeniser` in `tokenisers.py`
2. Implement required methods and add it to the `REGISTRY` dictionary

### Adding a New Embedding Type

1. Create a new class that inherits from `BaseEmbedding` in `embeddings.py`
2. Implement required methods and add it to the `REGISTRY` dictionary

### Adding a New Encoder Architecture

1. Create a new class that inherits from `BaseTower` in `encoders.py`
2. Implement required methods and add it to the `TOWER_REGISTRY` dictionary

### Adding a New Search Implementation

1. Create a new class that inherits from `BaseSearch` in `inference/search/base.py`
2. Implement required methods

## Documentation

- [Configuration Reference](artifacts/docs/config.md)
- [Evaluation Metrics](artifacts/docs/evaluation.md)
- [Refactoring & Architecture](artifacts/docs/refactoring.md)
- [Inference & Search](artifacts/docs/inference.md)






Plan out pipeline: 




Plan out data model: 



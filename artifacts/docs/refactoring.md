# Refactoring & Architecture

This document explains the restructuring of the Two-Tower codebase to improve modularity, separation of concerns, and extensibility.

## Key Goals

The refactoring had several main goals:

1. **Modularity**: Break down the monolithic implementation into small, well-defined components
2. **Separation of Concerns**: Separate training code from inference functionality
3. **Common Interfaces**: Create consistent interfaces for different implementations
4. **Extensibility**: Make it easy to add new components without changing existing code

## Directory Structure

The new directory structure clearly separates training from inference:

```
two-towers/
  ├── twotower/              # Core model training code
  │   ├── tokenisers.py      # Tokenization (Stage 1)
  │   ├── embeddings.py      # Embedding layers (Stage 2)
  │   ├── encoders.py        # Encoder architectures (Stage 3)
  │   ├── losses.py          # Loss functions (Stage 4)
  │   ├── dataset.py         # Dataset handling
  │   ├── train.py           # Training orchestration
  │   ├── utils.py           # Utility functions
  │   └── evaluate.py        # Evaluation metrics
  │
  ├── inference/             # Inference and retrieval code
  │   ├── search/            # Search implementations
  │   │   ├── base.py        # Base search interface
  │   │   ├── glove.py       # GloVe-based search
  │   │   └── two_tower.py   # Two-Tower model search
  │   ├── cli/               # Command-line tools
  │   │   └── retrieve.py    # Document retrieval CLI
  │   └── examples/          # Example scripts
  │
  ├── configs/               # Configuration files
  ├── docs/                  # Documentation
  ├── tools/                 # Utility scripts
  └── artifacts/             # Project artifacts and documentation
```

## Key Benefits

### 1. Cleaner Separation of Concerns

- **Training**: The `twotower` package focuses solely on model training and evaluation
- **Inference**: The `inference` package handles all retrieval and search functionality
- **Configuration**: YAML files in `configs/` manage all settings
- **Documentation**: Documentation for each component is clearly organized

### 2. Unified Interfaces

- **BaseTokeniser**: Common interface for all tokenization methods
- **BaseEmbedding**: Common interface for all embedding implementations
- **BaseTower**: Common interface for all encoder architectures
- **BaseSearch**: Common interface for all search implementations

### 3. Improved Extensibility

Adding new components is straightforward:

1. Create a new class that inherits from the appropriate base class
2. Implement the required methods
3. Add it to the corresponding registry

No changes to existing code are needed to add new functionality.

### 4. Better CLI Tools

- Command-line tools now have clearer organization
- Each tool has a single responsibility
- Better argument parsing and help documentation

### 5. Comprehensive Evaluation

- Standard IR metrics for evaluating model performance
- Utilities for comparing different models and configurations

## The Five-Stage Pipeline

The code follows a 5-stage pipeline:

1. **Tokenization**: Converts text to token IDs
   - Implemented in `tokenisers.py`
   - Classes: `CharTokeniser`, `WordTokeniser`

2. **Embedding**: Maps token IDs to dense vectors
   - Implemented in `embeddings.py`
   - Classes: `LookupEmbedding`, `FrozenWord2Vec`, `GloVeEmbedding`

3. **Encoding**: Transforms token embeddings into a single vector
   - Implemented in `encoders.py`
   - Classes: `MeanPoolingTower`, `AveragePoolingTower`

4. **Loss Function**: Defines the training objective
   - Implemented in `losses.py`
   - Functions: `contrastive_triplet_loss`

5. **Training**: Orchestrates the training process
   - Implemented in `train.py`
   - Function: `train_model`

## Search Implementations

The search functionality is now separated into the `inference` package:

- **BaseSearch**: Abstract base class defining the search interface
- **GloVeSearch**: Search implementation using pre-trained GloVe embeddings
- **TwoTowerSearch**: Search implementation using a trained Two-Tower model

## Migration Path

To help users migrate from the old structure to the new one:

1. The `tools/migrate_to_inference.py` script identifies code using the old structure
2. The script suggests updates to imports and API calls
3. Documentation in `artifacts/docs/` explains the new structure and usage

## Example Update Required

Old imports:
```python
from twotower.glove_search import GloVeSearch
from twotower.search import SemanticSearch
```

New imports:
```python
from inference.search import GloVeSearch
from inference.search import TwoTowerSearch  # Renamed from SemanticSearch
```

## Command-Line Usage Changes

Old CLI:
```bash
python retrieve.py --model model.pt --index index.pkl --query "search query"
```

New CLI:
```bash
python -m inference.cli.retrieve search --model model.pt --index index.pkl --query "search query"
``` 
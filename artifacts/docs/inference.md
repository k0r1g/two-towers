# Inference & Search

This document provides information about the inference and search capabilities of the Two-Tower model.

## Overview

The Two-Tower model provides a modular inference framework for semantic search through the `inference` package. This package includes:

- A common interface for different search implementations
- Ready-to-use search implementations for GloVe and Two-Tower models
- Command-line tools for building indices and searching
- Example scripts demonstrating usage

## Directory Structure

```
inference/
  ├── search/              # Search implementations
  │   ├── base.py          # Base search interface
  │   ├── glove.py         # GloVe-based search
  │   └── two_tower.py     # Two-Tower model search
  ├── cli/                 # Command-line tools
  │   └── retrieve.py      # Document retrieval CLI
  └── examples/            # Example scripts
      ├── glove_search_example.py       # GloVe search example
      └── evaluate_model_example.py     # Model evaluation example
```

## Search Implementations

All search implementations extend the `BaseSearch` abstract class, which provides a common interface:

```python
class BaseSearch(ABC):
    @abstractmethod
    def index_documents(self, documents: List[str]) -> None:
        """Create and store embeddings for a list of documents."""
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Union[str, float]]]:
        """Search for documents similar to the query."""
        pass
    
    def save_index(self, filepath: str) -> None:
        """Save the document index to disk."""
        pass
    
    def load_index(self, filepath: str) -> None:
        """Load a document index from disk."""
        pass
```

### GloVe Search

The `GloVeSearch` class provides semantic search using pre-trained GloVe embeddings:

```python
from inference.search import GloVeSearch

# Initialize
glove_search = GloVeSearch(model_name='glove-wiki-gigaword-50')

# Index documents
documents = ["Doc 1 content", "Doc 2 content", "Doc 3 content"]
glove_search.index_documents(documents)

# Search
results = glove_search.search("my query", top_k=3)

# Save/load index
glove_search.save_index("glove_index.pkl")
glove_search.load_index("glove_index.pkl")
```

### Two-Tower Search

The `TwoTowerSearch` class provides semantic search using a trained Two-Tower model:

```python
from twotower import load_checkpoint
from inference.search import TwoTowerSearch

# Load model and tokenizer
checkpoint = load_checkpoint("model.pt")
model = checkpoint["model"]
tokenizer = checkpoint["tokenizer"]

# Initialize
tt_search = TwoTowerSearch(model, tokenizer, device="cuda")

# Index documents
documents = ["Doc 1 content", "Doc 2 content", "Doc 3 content"]
tt_search.index_documents(documents)

# Search
results = tt_search.search("my query", top_k=3)

# Save/load index
tt_search.save_index("model_index.pkl")
tt_search.load_index("model_index.pkl")
```

## Command-Line Interface

The CLI module provides tools for building search indices and retrieving documents:

### Building an Index

```bash
python -m inference.cli.retrieve build-index \
  --model path/to/model.pt \
  --documents path/to/docs.txt \
  --output path/to/index.pkl \
  --tokenizer char
```

### Searching Using an Index

```bash
python -m inference.cli.retrieve search \
  --model path/to/model.pt \
  --index path/to/index.pkl \
  --query "search query" \
  --top-k 5 \
  --tokenizer char
```

## Example Scripts

The `inference/examples/` directory contains example scripts demonstrating how to use the search functionality:

### GloVe Search Example

`glove_search_example.py` demonstrates how to use the `GloVeSearch` class:

```python
from inference.search import GloVeSearch

# Initialize GloVe search
search_engine = GloVeSearch(model_name='glove-wiki-gigaword-50')

# Index documents
docs = ["Doc 1", "Doc 2", "Doc 3"]
search_engine.index_documents(docs)

# Search
results = search_engine.search("query", top_k=3)
```

### Evaluate Model Example

`evaluate_model_example.py` demonstrates how to evaluate a model's search performance:

```python
from twotower.evaluate import evaluate_model, print_evaluation_results

# Load model and test data
# ...

# Evaluate
results = evaluate_model(
    model=model,
    test_data=test_data,
    tokenizer=tokenizer
)

# Print results
print_evaluation_results(results)
```

## Extending with New Search Implementations

To add a new search implementation:

1. Create a new class that inherits from `BaseSearch` in `inference/search/base.py`
2. Implement the required methods: `index_documents()`, `search()`, and optionally `save_index()` and `load_index()`
3. Add your implementation to the `__init__.py` file

For example:

```python
from .base import BaseSearch

class MyCustomSearch(BaseSearch):
    def __init__(self, custom_param):
        self.custom_param = custom_param
        self.documents = None
        self.document_embeddings = None
    
    def index_documents(self, documents):
        # Implementation
        pass
    
    def search(self, query, top_k=5):
        # Implementation
        pass
``` 
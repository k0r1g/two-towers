# Inference Module for Two-Tower Models

This module provides inference capabilities for Two-Tower models, including semantic search and retrieval functionalities.

## Structure

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

## Usage

### Search

The search module provides a unified interface for different search implementations:

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

### Command-line Interface

The CLI module provides tools for building search indices and retrieving documents:

```bash
# Build an index
python -m inference.cli.retrieve build-index --model path/to/model.pt --documents path/to/docs.txt --output path/to/index.pkl

# Search using an index
python -m inference.cli.retrieve search --model path/to/model.pt --index path/to/index.pkl --query "search query"
```

### Evaluation

The `twotower.evaluate` module provides evaluation metrics for retrieval models:

```python
from twotower.evaluate import evaluate_model, print_evaluation_results

results = evaluate_model(
    model=model,
    test_data=test_data,
    tokenizer=tokenizer,
    metrics=['precision', 'recall', 'mrr', 'ndcg'],
    k_values=[1, 3, 5]
)

print_evaluation_results(results)
``` 
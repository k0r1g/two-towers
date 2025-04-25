# Model Evaluation

The Two-Tower model includes comprehensive evaluation metrics for information retrieval tasks. The evaluation module in `twotower.evaluate` provides both standard metrics and utilities for model assessment.

## Available Metrics

- **Precision@K**: Proportion of relevant documents in the top-K retrieved results
- **Recall@K**: Proportion of all relevant documents that were retrieved in the top-K
- **Mean Reciprocal Rank (MRR)**: Average of the reciprocal of the rank of the first relevant document
- **Normalized Discounted Cumulative Gain (NDCG@K)**: Measures ranking quality considering relevance scores and position

## Usage

### Basic Evaluation

```python
from twotower.evaluate import evaluate_model, print_evaluation_results

# Test data format: List of (query, documents, relevance_scores) tuples
test_data = [
    (
        "machine learning algorithm", 
        [
            "Machine learning algorithms are used in AI applications.",
            "Deep learning is a subset of machine learning.",
            "Python is a programming language used for data science.",
            # more documents...
        ],
        [1, 1, 0, 0, 1]  # Relevance scores (1 = relevant, 0 = irrelevant)
    ),
    # more test examples...
]

# Evaluate model
results = evaluate_model(
    model=model,
    test_data=test_data,
    tokenizer=tokenizer,
    metrics=['precision', 'recall', 'mrr', 'ndcg'],
    k_values=[1, 3, 5, 10],
    batch_size=32,
    device='cuda'
)

# Print formatted results
print_evaluation_results(results)
```

### Individual Metrics

You can also use individual metric functions:

```python
from twotower.evaluate import precision_at_k, recall_at_k, mean_reciprocal_rank, ndcg_at_k

# Relevance scores for a query (1 = relevant, 0 = irrelevant)
relevance = [1, 0, 1, 0, 0, 1]

# Calculate metrics
p_at_3 = precision_at_k(relevance, k=3)
r_at_3 = recall_at_k(relevance, k=3, total_relevant=3)
mrr = mean_reciprocal_rank(relevance)
ndcg = ndcg_at_k(relevance, k=5)
```

## Evaluating Multiple Models

To compare multiple models:

```python
models = {
    "char_cnn": char_cnn_model,
    "word_avg": word_avg_model,
    "glove": glove_model
}

results = {}
for name, model in models.items():
    results[name] = evaluate_model(
        model=model,
        test_data=test_data,
        tokenizer=tokenizer
    )

# Compare results
for name, metrics in results.items():
    print(f"\nResults for {name}:")
    print_evaluation_results(metrics)
```

## Custom Evaluation Datasets

For custom evaluation datasets, you can create a loader function:

```python
def load_test_data(filepath):
    """Load test data from a file."""
    test_data = []
    # Parse file and create (query, documents, relevance) tuples
    # ...
    return test_data

# Load and evaluate
test_data = load_test_data("path/to/test_data.json")
results = evaluate_model(model, test_data, tokenizer)
``` 
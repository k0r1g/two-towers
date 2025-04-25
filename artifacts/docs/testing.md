# Testing Framework

This document describes the testing framework for the Two-Tower model.

## Running Tests

To run all tests:

```bash
python tests/run_tests.py
```

To run a specific test file:

```bash
python -m unittest tests/search/test_glove_search.py
```

## Test Structure

- `tests/search/`: Tests for search implementations
  - `test_glove_search.py`: Unit tests for GloVe search functionality
  - `benchmark_glove_search.py`: Performance benchmarks for GloVe search

## Benchmarks

The benchmark scripts measure performance metrics and generate reports in the `reports/` directory. To run the GloVe search benchmark:

```bash
python tests/search/benchmark_glove_search.py
```

This will generate:
- CSV files with timing data
- PNG files with performance graphs

## Adding New Tests

When adding new tests:

1. Create a new file with the pattern `test_*.py`
2. Implement test cases using the `unittest` framework
3. Add your test file to the appropriate directory (e.g., `tests/search/` for search-related tests)

## Environment Setup

Tests assume that the required dependencies are installed, including:
- gensim (for GloVe embeddings)
- numpy
- scikit-learn
- matplotlib (for benchmarks) 
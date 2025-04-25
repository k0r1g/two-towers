# Two-Tower Model Documentation

Welcome to the documentation for the Two-Tower retrieval model. This documentation provides detailed information about the model architecture, configuration, evaluation metrics, and usage.

## Table of Contents

- [Configuration Reference](config.md) - Complete reference for all configuration options
- [Evaluation Metrics](evaluation.md) - Information about model evaluation metrics
- [Inference & Search](inference.md) - Guide to using the search functionality
- [Refactoring & Architecture](refactoring.md) - Overview of the codebase structure and design

## Quick Links

- [GitHub Repository](https://github.com/yourusername/two-towers)
- [Main README](../../README.md)

## Overview

The Two-Tower retrieval model is a dual encoder architecture for semantic search and document retrieval. The model consists of two "towers" - one for queries and one for documents - that encode text into dense vector representations. These representations can then be compared using cosine similarity to find the most relevant documents for a given query.

## Features

- **Modular Design**: Each component (tokenization, embedding, encoding) is implemented as a separate module
- **Config-Driven**: All model and training parameters defined in YAML configuration files
- **Easily Extensible**: Adding new tokenizers, embeddings, or encoders only requires implementing a new class
- **Standard IR Metrics**: Comprehensive evaluation metrics (Precision@K, Recall@K, MRR, NDCG)
- **Unified Search Interface**: Common interface for different search implementations
- **CLI Tools**: Command-line tools for building indices and retrieving documents

## Getting Started

See the [main README](../../README.md) for instructions on installation and basic usage. 
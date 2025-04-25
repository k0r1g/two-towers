# Two-Tower Retrieval Model Documentation

Welcome to the documentation for the Two-Tower Retrieval Model. This documentation provides comprehensive information about the model's architecture, configuration, training, evaluation, and inference capabilities.

## Table of Contents

- [Project Status](../project_status_report.md): Current state, recent developments, and next steps
- [Configuration](config.md): Configuration options and usage
- [Refactoring & Architecture](refactoring.md): Information about the codebase structure
- [Inference & Search](inference.md): Using the model for retrieval tasks
- [Evaluation](evaluation.md): Metrics and evaluation methodologies
- [Testing](testing.md): Testing framework and procedures
- [GPU Setup](gpu-setup.md): Setting up GPU environment for training
- [Docker Setup](docker-setup.md): Containerized deployment for inference
- [HuggingFace Hub Integration](huggingface.md): Sharing models via HuggingFace Hub

## Quick Links

- [GitHub Repository](https://github.com/yourusername/two-towers)
- [Main README](../../README.md)
- [HuggingFace Hub](https://huggingface.co/models?search=mlx7-two-tower)

## Overview

The Two-Tower retrieval model is a dual encoder architecture for semantic search and document retrieval. The model consists of two "towers" - one for queries and one for documents - that encode text into dense vector representations. These representations can then be compared using cosine similarity to find the most relevant documents for a given query.

## Features

- **Modular Design**: Each component (tokenization, embedding, encoding) is implemented as a separate module
- **Config-Driven**: All model and training parameters defined in YAML configuration files
- **Easily Extensible**: Adding new tokenizers, embeddings, or encoders only requires implementing a new class
- **Standard IR Metrics**: Comprehensive evaluation metrics (Precision@K, Recall@K, MRR, NDCG)
- **Unified Search Interface**: Common interface for different search implementations
- **CLI Tools**: Command-line tools for building indices and retrieving documents
- **GPU Acceleration**: Support for CUDA-enabled training and inference
- **HuggingFace Integration**: Share and load models from HuggingFace Hub

## Getting Started

To get started with the Two-Tower model, please refer to the [README.md](../../README.md) file in the project root, which provides installation instructions and basic usage examples. For setting up a GPU environment, see the [GPU Setup guide](gpu-setup.md).

## Configuration System

The Two-Tower model uses a hierarchical YAML-based configuration system. For details, see the [Configuration documentation](config.md).

## Inference

For information about using the trained model for search and retrieval tasks, see the [Inference documentation](inference.md).

## Sharing Models

The project supports sharing models via the HuggingFace Hub. To learn how to share your trained models, see the [HuggingFace Hub Integration documentation](huggingface.md).

## Contributing

If you're interested in contributing to the project, please review the [Refactoring & Architecture documentation](refactoring.md) to understand the codebase structure.

## Testing

For information about the testing framework and how to run tests, see the [Testing documentation](testing.md). 
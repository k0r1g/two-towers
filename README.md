---
language: en
license: mit
datasets:
- microsoft/ms_marco
tags:
- word2vec
- cbow
- continuous bag of words
- embedding
---

# MS MARCO Word2Vec Embedding Model

This repository contains a Continuous Bag of Words (CBOW) Word2Vec model trained on the Microsoft MS MARCO dataset.

## Model Details

- **Architecture**: CBOW (Continuous Bag of Words)
- **Embedding Dimension**: 128
- **Context Window Size**: 4
- **Vocabulary Size**: 50,001
- **Training Pairs**: 6,618,785
- **Parameters**: 12,800,256
- **Training Device**: cuda

## Usage

```python
import torch

# Load the model
vocab_size = 50001
embed_dim = 128
model = CBOW(vocab_size=vocab_size, embed_dim=embed_dim)
model.load_state_dict(torch.load("cbow_model.pth"))

# Get embeddings for words
embeddings = model.embeddings.weight  # Shape: [vocab_size, embed_dim]
```

## Training

This model was trained for 5 epochs with a batch size of 256 and learning rate of 0.003.

## License

MIT

# MS MARCO Search Engine — Engineering Pipeline

> This document captures the end‑to‑end pipeline that powers our two‑tower search engine, maps each conceptual block in the architecture diagram to the concrete scripts in this repo, and describes how the system will evolve over **three iterations**.

```mermaid
flowchart LR
    subgraph Data & Pre‑Processing
        A[Raw MS MARCO v1.1\n(5000 queries)] -->|clean / select cols| B[00_train_tkn.py\nCustom Tokeniser]
        B -->|tokens → ids| C[word_to_idx.pkl\nquery/relevant/irrelevant_*.pkl]
        C --> D[01_train_w2v.py\nCBOW Word2Vec\nemb_dim = 128]
    end
    subgraph Model Training
        D -->|load weights| E[Embedding Layer\n(frozen or fine‑tuned)]
        E --> F[02_train_dualen.py\nQuery Tower\nfc(100→100)]
        E --> G[02_train_dualen.py\nDocument Tower\nfc(100→100)]
        F --> H[Triplet Loss (margin=0.2)]
        G --> H
    end
```

---

## Problem Statement
Given a user **query**, retrieve the top‑k documents ranked by semantic relevance. We train a **two‑tower** neural network so that the distance between the query encoding and *relevant* document encodings is small, and the distance to *irrelevant* document encodings is large. Training is performed with **triplet loss**.

---

## Block‑by‑Block Details

### 1️⃣ Dataset (MS MARCO v1.1)  
**Script(s):** none (Hugging Face `datasets` inside `00_train_tkn.py`)  
**Rows used (iteration 1):** first `NUM_EXAMPLES = 5000` *queries* (≈ 82 k passages)  
**Columns consumed:**
- `query` (string) — the user search query
- `passages.passage_text` (list[string]) — candidate passages
- `passages.is_selected` (list[int]) — indicator of which passage was clicked

**Positive / negative definition (iteration 1)**  
All passages *within the same row* are treated as **relevant** to that row's query, ignoring `is_selected`. Passages belonging to *other* queries are treated as **irrelevant**.  
*Why?* This gives many easy positive examples quickly without building a click‑model; good for a first pass.  
*Future options:* use `is_selected` for hard‑positive/negative mining, or sample "hard negatives" that share tokens with the query.

**Planned data re‑packaging**  
We will materialise a new Parquet dataset where **one row = (query, positive_passage, negative_passage)** to speed up future pipelines and enable Spark‑style processing.

---

### 2️⃣ Tokeniser — `00_train_tkn.py`
| Aspect | Value |
|---|---|
| Tokenisation pipeline | lowercase → regex clean → NLTK `word_tokenize` |
| Stop‑words removed? | yes (`nltk.corpus.stopwords`) |
| Vocabulary size | `TOP_N_WORDS = 50 000` + `<PAD>` |
| Saved artifacts | `word_to_idx.pkl`, `idx_to_word.pkl`, token‑ID lists for queries / passages |
| Tracking | Weights & Biases run `ms-marco-tokenizer` |

*Future improvements*
- Replace NLTK with a learned BPE / SentencePiece model
- Export a `datasets` `Dataset` object for seamless streaming
- Support sub‑word units for OOV handling

---

### 3️⃣ Word2Vec Embedding — `01_train_w2v.py`
| Hyper‑parameter | Value |
|---|---|
| Architecture | CBOW |
| `EMBED_DIM` | **128** |
| Context window | **4** tokens on each side |
| Batch size | 256 |
| Epochs | 5 |
| Optimiser | Adam (`lr=3e‑3`) |

**Outputs**  
`./checkpoints/cbow_<timestamp>_final.pth` — weights for `torch.nn.Embedding`.  
Model card + vocab uploaded to HF hub.

**Usage downstream**  
Weights are loaded into an `nn.Embedding` inside `02_train_dualen.py`. They are **frozen** for iteration 1; we may unfreeze for fine‑tuning in later iterations.

---

### 4️⃣ Dual‑Encoder (Two‑Tower) — `02_train_dualen.py`
| Component | Shape / Details |
|---|---|
| Embedding layer | `nn.Embedding(vocab_size, 100)` (*weights copied* from Word2Vec and truncated/padded if needed) |
| Query Tower | `nn.Linear(100 → 100)` + ReLU (implicitly via cosine) |
| Document Tower | identical structure, *weights **not** shared* |
| Similarity | Cosine similarity (`F.cosine_similarity`) |
| Loss | Triplet loss: `max(0, margin – (sim_pos – sim_neg))`, `margin = 0.2` |
| Optimiser | Adam (`lr = 1e‑3`) over both towers |
| Batch size | 32 triples |

**Grad‑flow**  
Back‑prop updates only the tower layers (embeddings are frozen).  
`embedding_layer.weight.requires_grad` can be toggled to unfreeze.

---

## Inference Path
1. **Offline** — encode every document once with the trained **Document Tower**, store as float32 matrix.
2. **Online** — for an incoming query: tokenise → embed → **Query Tower** → cosine similarity against cached doc encodings → return top‑k (k = 5).

---

## Road‑map of Iterations
| Iteration | Dataset | Tokeniser | Embedding | Encoder Towers | Loss & Margin | Notes |
|---|---|---|---|---|---|---|
| **1** (MVP) | v1.1 – 5 k queries | NLTK word; 50 k vocab | CBOW, 128‑d, **frozen** | Mean‑pool + Linear(100) | Triplet, 0.2 | Quick turnaround |
| **2** | v1.1 – **full** 102 k queries | same script | same W2V **fine‑tune** | same towers | same | Stress‑test scalability |
| **3** | v1.2 | Possibly SentencePiece | CBOW re‑trained 200‑d | **GRU( hidden=256 )** per tower → final hidden | Triplet, tune margin | Hard‑negative mining, multi‑GPU |

---

## Open Questions / TODOs
- [ ] Consolidate duplicate embedding‑dim values (128 vs 100).
- [ ] Move `QryTower` / `DocTower` classes from `02_train_dualen.py` into `model.py` and import them to keep code DRY.
- [ ] Implement **Parquet triple dataset** builder.
- [ ] Evaluate retrieval quality (MRR@10) on MS MARCO dev set.
- [ ] Add FastAPI endpoint for real‑time inference.

---

*Last updated:* <!--TIMESTAMP-->



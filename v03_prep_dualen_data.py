#!/usr/bin/env python
"""
03_prep_dualen_data.py
-----------------------
Prepares train, val, and test datasets for the dual encoder model by:
- Building triplets for val/test
- Tokenizing all splits using the train vocab
- Embedding with pretrained Word2Vec
- Average pooling
- Saving PyTorch-ready datasets
"""

import os, pickle, json
import torch
from pathlib import Path
from v00_build_triplets import build_triplets
from v01_train_tkn import preprocess as tokenize
from gensim.models import Word2Vec
from torch.utils.data import Dataset
from dataset import DualEncoderDataset

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SPLITS_TO_TOKENIZE = ["train", "val", "test"]
TRIPLET_PATH = Path("data/triplets")
TOKEN_PATH   = Path("data/tokens")
FINAL_PATH   = Path("data/dualen")
TRAIN_TOKEN_DIR = TOKEN_PATH / "train"
WORD_TO_IDX = TRAIN_TOKEN_DIR / "word_to_idx.pkl"
EMBEDDING_MODEL = "checkpoints/w2v_cbow.model"  

# ─────────────────────────────────────────────────────────────────────────────
# 1️⃣ Build triplets for val/test (train is assumed prebuilt)
# ─────────────────────────────────────────────────────────────────────────────
for split in ["val", "test"]:
    jsonl_path = TRIPLET_PATH / f"{split}_triplets.jsonl"
    if not jsonl_path.exists():
        print(f"🔧 Building {split} triplets...")
        build_triplets(version="v1.1", split=split,
                       num_examples=None, outfile=jsonl_path)
    else:
        print(f"✅ Found existing {split} triplets at {jsonl_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 2️⃣ Load vocab from train set
# ─────────────────────────────────────────────────────────────────────────────
with open(WORD_TO_IDX, "rb") as f:
    word_to_idx = pickle.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# 3️⃣ Tokenize triplets for all splits using train vocab
# ─────────────────────────────────────────────────────────────────────────────
def tokenize_triplets(split):
    in_file = TRIPLET_PATH / f"{split}_triplets.jsonl"
    out_dir = TOKEN_PATH / split
    out_dir.mkdir(parents=True, exist_ok=True)

    q_ids, p_ids, n_ids = [], [], []

    with open(in_file, encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            q = [word_to_idx.get(w, 0) for w in tokenize(j["query"])]
            p = [word_to_idx.get(w, 0) for w in tokenize(j["positive"])]
            n = [word_to_idx.get(w, 0) for w in tokenize(j["negative"])]
            q_ids.append(q); p_ids.append(p); n_ids.append(n)

    for name, val in zip(["query", "positive", "negative"], [q_ids, p_ids, n_ids]):
        with open(out_dir / f"{name}_ids.pkl", "wb") as f:
            pickle.dump(val, f)
    print(f"✅ Tokenized {split} → {out_dir}")

for split in SPLITS_TO_TOKENIZE:
    tokenize_triplets(split)

# ─────────────────────────────────────────────────────────────────────────────
# 4️⃣ Load Word2Vec model
# ─────────────────────────────────────────────────────────────────────────────
print("🔍 Loading Word2Vec model...")
w2v = Word2Vec.load(EMBEDDING_MODEL)

def embed_and_average(data_dir):
    with open(data_dir / "query_ids.pkl", "rb") as f:
        queries = pickle.load(f)
    with open(data_dir / "positive_ids.pkl", "rb") as f:
        positives = pickle.load(f)
    with open(data_dir / "negative_ids.pkl", "rb") as f:
        negatives = pickle.load(f)

    def embed(seq):
        vectors = [w2v.wv[w2v.wv.index2word[i]] for i in seq if i < len(w2v.wv)]
        return torch.tensor(vectors).float().mean(dim=0) if vectors else torch.zeros(w2v.vector_size)

    triplets = []
    for q, p, n in zip(queries, positives, negatives):
        triplets.append((embed(q), embed(p), embed(n)))
    return triplets

# ─────────────────────────────────────────────────────────────────────────────
# 5️⃣ Save averaged dual encoder dataset
# ─────────────────────────────────────────────────────────────────────────────
FINAL_PATH.mkdir(parents=True, exist_ok=True)

for split in SPLITS_TO_TOKENIZE:
    token_dir = TOKEN_PATH / split
    print(f"🔄 Embedding + pooling for {split}...")
    triplets = embed_and_average(token_dir)
    dataset = DualEncoderDataset(triplets)
    torch.save(dataset, FINAL_PATH / f"{split}_dualen.pt")
    print(f"✅ Saved final {split} dataset to {FINAL_PATH}/{split}_dualen.pt")

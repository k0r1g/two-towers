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
from dotenv import load_dotenv
import wandb
import datetime
import sys
import torch
from pathlib import Path
from v00_build_triplets import build_triplets
from v01_train_tkn import preprocess as tokenize
from torch.utils.data import Dataset
from dataset import DualEncoderDataset
from model import CBOW            # CBOW embedding model
import glob                       # to pick the latest checkpoint

# Optional HF upload
try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("📦 Installing huggingface_hub...")
    os.system(f"{sys.executable} -m pip install -q huggingface_hub")
    from huggingface_hub import HfApi, create_repo

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SPLITS_TO_TOKENIZE = ["train", "val", "test"]
TRIPLET_PATH = Path("data/triplets")
TOKEN_PATH   = Path("data/tokens")
FINAL_PATH   = Path("data/dualen")
TRAIN_TOKEN_DIR = TOKEN_PATH / "train"
WORD_TO_IDX = TRAIN_TOKEN_DIR / "word_to_idx.pkl"
# (Word2Vec constant removed – using CBOW checkpoints instead)
EMBEDDING_MODEL = "checkpoints/w2v_cbow.model"  

# ─────────────────────────────────────────────────────────────────────────────
# Additional logging configuration (HF & wandb)
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

HF_REPO_ID   = "Kogero/msmarco-dualen-data"
HF_REPO_TYPE = "dataset"
WANDB_PROJECT = "msmarco-dualen-prep"
RUN_NAME      = f"dualen-prep-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

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
# 4️⃣  Load trained CBOW embedding weights
# ─────────────────────────────────────────────────────────────────────────────
print("🔍 Loading CBOW model checkpoint …")

# Find the newest *_final.pth – or fall back to any .pth file
ckpts = sorted(glob.glob("checkpoints/cbow_*_final.pth")) or sorted(glob.glob("checkpoints/cbow_*.pth"))
assert ckpts, "❌ No CBOW checkpoints found in ./checkpoints"
ckpt_path = ckpts[-1]
print(f"   ↳ using checkpoint {ckpt_path}")

# Re‑instantiate the model skeleton and load weights
state_dict = torch.load(ckpt_path, map_location="cpu")
vocab_size = state_dict["emb.weight"].shape[0]
embed_dim  = state_dict["emb.weight"].shape[1]

cbow = CBOW(vocab_size=vocab_size, embed_dim=embed_dim)
cbow.load_state_dict(state_dict)
cbow.eval()

# Freeze and take the raw embedding weight matrix
embedding_weight = cbow.emb.weight.detach()           # Tensor [vocab_size, embed_dim]

def get_embedding(seq, average=True):
    # Keep only valid token IDs
    valid = [i for i in seq if 0 <= i < embedding_weight.shape[0]]
    if not valid:
        return (
            torch.zeros(embed_dim)               if average
            else torch.zeros((1, embed_dim))
        )

    vectors = embedding_weight[valid]            # (seq_len, embed_dim)
    return (
        vectors.mean(dim=0)                      if average
        else vectors
    )

def embed_triplets(data_dir, average=True):
    with open(data_dir / "query_ids.pkl", "rb") as f:
        queries = pickle.load(f)
    with open(data_dir / "positive_ids.pkl", "rb") as f:
        positives = pickle.load(f)
    with open(data_dir / "negative_ids.pkl", "rb") as f:
        negatives = pickle.load(f)

    triplets = []
    for q, p, n in zip(queries, positives, negatives):
        triplets.append((
            get_embedding(q, average),
            get_embedding(p, average),
            get_embedding(n, average)
        ))
    return triplets

# ─────────────────────────────────────────────────────────────────────────────
# Start wandb run
# ─────────────────────────────────────────────────────────────────────────────
run = wandb.init(
    project=WANDB_PROJECT,
    name=RUN_NAME,
    config={
        "vocab_size": len(word_to_idx),
        "embedding_dim": embed_dim,
        "splits": SPLITS_TO_TOKENIZE,
    }
)

# ─────────────────────────────────────────────────────────────────────────────
# 5️⃣ Save averaged dual encoder dataset
# ─────────────────────────────────────────────────────────────────────────────
FINAL_PATH.mkdir(parents=True, exist_ok=True)

for split in SPLITS_TO_TOKENIZE:
    token_dir = TOKEN_PATH / split
    FINAL_PATH.mkdir(parents=True, exist_ok=True)

    # ── Save average-pooled version (MLP-ready)
    print(f"🔄 [MLP] Embedding + pooling for {split}...")
    pooled_triplets = embed_triplets(token_dir, average=True)
    torch.save(pooled_triplets, FINAL_PATH / f"{split}_dualen_avg.pt")
    print(f"✅ Saved avg-pooled {split} to ➜ {FINAL_PATH}/{split}_dualen_avg.pt")

    # Log to wandb
    run.log({f"{split}_num_triplets": len(pooled_triplets)})

    # ── Save full-sequence version (RNN-ready)
    print(f"🔄 [RNN] Embedding full sequences for {split}...")
    unpooled_triplets = embed_triplets(token_dir, average=False)
    torch.save(unpooled_triplets, FINAL_PATH / f"{split}_dualen_seq.pt")
    print(f"✅ Saved full-seq {split} to ➜ {FINAL_PATH}/{split}_dualen_seq.pt")

# ─────────────────────────────────────────────────────────────────────────────
# Upload artefacts to Hugging Face Hub (optional)
# ─────────────────────────────────────────────────────────────────────────────

hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HUGGINGFACE_KEY")
if hf_token:
    print(f"🚀 Uploading to Hugging Face: {HF_REPO_ID}")
    api = HfApi()
    create_repo(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, token=hf_token, exist_ok=True)

    # Upload each dualen .pt file
    for pt_file in FINAL_PATH.glob("*_dualen_*.pt"):
        print(f"  ↳ Uploading {pt_file.name} to HF Hub…")
        api.upload_file(
            path_or_fileobj=str(pt_file),
            path_in_repo=pt_file.name,
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            token=hf_token
        )

    print(f"✅ Upload complete: https://huggingface.co/datasets/{HF_REPO_ID}")
else:
    print("⚠️  HUGGINGFACE_TOKEN not set – skipping HF upload.")

# ─────────────────────────────────────────────────────────────────────────────
# Final cleanup
# ─────────────────────────────────────────────────────────────────────────────
run.finish()
print("📊 wandb logging complete.")

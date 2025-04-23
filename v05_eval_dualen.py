#!/usr/bin/env python
"""
05_eval_dualen.py
-----------------
Inference + evaluation for the dual-encoder:

1.  Load trained QryTower & DocTower checkpoints.
2.  Encode *every unique* document vector from the test set **once** and store
    the resulting embeddings in ChromaDB.
3.  For each query in the test split:
      • encode the query
      • retrieve top-k (k=5) most-similar docs from Chroma
      • compute triplet-margin loss
      • compute precision@5, recall@5, and F1@5
4.  Report average loss + F1@5.

NOTE
────
We bypass Chroma's internal embedding model: we supply **our own embeddings**
via the `embeddings=` argument when we `add()` documents and
`query_embeddings=` when we search.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
from pathlib import Path
import pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm

# 3-rd-party vector DB
import chromadb
from chromadb.utils import embedding_functions  # (needed for type hints only)

# Local modules
from model import QryTower, DocTower

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
EMB_DIM      = 128
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR     = Path("data/dualen")
CKPT_PATH    = Path("checkpoints/dualen/dualen_epoch5.pt")  # last checkpoint
K            = 5
MARGIN       = 0.2
COLLECTION   = "dualen_test_cache"

# ─────────────────────────────────────────────────────────────────────────────
# Helper: triplet-margin loss (same as training)
# ─────────────────────────────────────────────────────────────────────────────
def triplet_margin_loss(q, pos, neg, margin=MARGIN):
    sim_pos = F.cosine_similarity(q, pos)
    sim_neg = F.cosine_similarity(q, neg)
    return torch.clamp(margin - (sim_pos - sim_neg), min=0.0)

# ─────────────────────────────────────────────────────────────────────────────
# 1️⃣  Load data & models
# ─────────────────────────────────────────────────────────────────────────────
print("📦 Loading test dataset and model checkpoint …")
test_ds = torch.load(DATA_DIR / "test_dualen.pt")

chkpt = torch.load(CKPT_PATH, map_location="cpu")
qry_enc = QryTower(EMB_DIM).to(DEVICE)
doc_enc = DocTower(EMB_DIM).to(DEVICE)
qry_enc.load_state_dict(chkpt["query_encoder"])
doc_enc.load_state_dict(chkpt["doc_encoder"])
qry_enc.eval(), doc_enc.eval()

# ─────────────────────────────────────────────────────────────────────────────
# 2️⃣  Build Chroma cache of *unique* document encodings
# ─────────────────────────────────────────────────────────────────────────────
print("🔐 Building / loading Chroma collection …")
client = chromadb.Client()                                  # in-memory
coll   = client.get_or_create_collection(name=COLLECTION)

if coll.count() == 0:
    print("🗂  Encoding and caching documents …")
    # Use tensor.bytes() as a hash to de-duplicate
    raw_vecs = {}
    for item in tqdm(test_ds, desc="Index docs"):
        for field in ("positive", "negative"):
            key = item[field].cpu().numpy().tobytes()
            if key not in raw_vecs:
                raw_vecs[key] = item[field]

    # Batch-encode with DocTower
    ids, embs, vec2id = [], [], {}          # vec2id  ➜  bytes → doc_id
    raw_items = list(raw_vecs.items())
    BATCH = 512
    for i in tqdm(range(0, len(raw_items), BATCH), desc="DocTower"):
        chunk_keys, chunk_vecs = zip(*raw_items[i : i + BATCH])
        encs = doc_enc(torch.stack(chunk_vecs).to(DEVICE)).cpu()
        for key, enc in zip(chunk_keys, encs):
            did = f"doc_{len(ids)}"
            vec2id[key] = did
            ids.append(did)
            embs.append(enc.tolist())

    coll.add(ids=ids, embeddings=embs)
    print(f"✅ Cached {len(ids):,} unique docs")
else:
    print(f"✅ Re-using existing cache with {coll.count():,} docs")
    # Build vec2id lazily from collection metadata if needed
    vec2id = {}  # (not used when cache already exists in this simple script)

# ─────────────────────────────────────────────────────────────────────────────
# 3️⃣  Iterate queries, gather metrics
# ─────────────────────────────────────────────────────────────────────────────
print("🔎 Running evaluation …")
tot_loss, tot_f1, n_queries = 0.0, 0.0, len(test_ds)

for item in tqdm(test_ds):
    # ── Encode query once
    q_vec = item["query"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        q_enc = qry_enc(q_vec).cpu()  # (1, dim)

    # ── Triplet loss (use pre-encoded p_enc / n_enc only for loss)
    with torch.no_grad():
        p_enc = doc_enc(item["positive"].unsqueeze(0).to(DEVICE)).cpu()
        n_enc = doc_enc(item["negative"].unsqueeze(0).to(DEVICE)).cpu()

    tot_loss += triplet_margin_loss(q_enc, p_enc, n_enc).item()

    # ── Retrieve top-k via Chroma
    retrieved_ids = coll.query(
        query_embeddings=q_enc.tolist(),
        n_results=K,
        include=["ids"]
    )["ids"][0]

    # ── Check hit  (vec2id gives us the cached ID of the true positive)
    pos_id  = vec2id[item["positive"].cpu().numpy().tobytes()]
    hit     = 1 if pos_id in retrieved_ids else 0

    precision = hit / K
    recall    = hit / 1
    f1        = 0.0 if hit == 0 else 2 * precision * recall / (precision + recall)
    tot_f1   += f1
# ─────────────────────────────────────────────────────────────────────────────
# 4️⃣  Results
# ─────────────────────────────────────────────────────────────────────────────
avg_loss = tot_loss / n_queries
avg_f1   = tot_f1   / n_queries

print("\n📊  EVALUATION SUMMARY")
print(f"   • Avg Triplet Loss : {avg_loss:.4f}")
print(f"   • Avg  F1@{K}       : {avg_f1:.4f}")

"""
You can persist the Chroma DB by using chromadb.PersistentClient()
pointing at a directory, if desired.
"""

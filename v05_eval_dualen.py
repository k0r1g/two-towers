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
    # Gather unique docs: we use stringified tensor bytes as key
    cache, id2vec = {}, {}
    for item in tqdm(test_ds, desc="Index docs"):
        # positive
        key = item["positive"].cpu().numpy().tobytes()
        if key not in cache:
            cache[key] = item["positive"]
        # negative
        key = item["negative"].cpu().numpy().tobytes()
        if key not in cache:
            cache[key] = item["negative"]

    # Encode with DocTower in batches for speed
    ids, embs = [], []
    all_docs  = list(cache.items())
    BATCH = 512
    for i in tqdm(range(0, len(all_docs), BATCH), desc="DocTower"):
        chunk = all_docs[i : i + BATCH]
        vecs  = torch.stack([v for _, v in chunk]).to(DEVICE)
        enc   = doc_enc(vecs).cpu()
        for (raw_id, _), e in zip(chunk, enc):
            str_id = f"doc_{len(ids)}"
            ids.append(str_id)
            embs.append(e.tolist())
            id2vec[str_id] = e           # keep for quick gt comparison

    # Store in Chroma (we can leave documents=None)
    coll.add(ids=ids, embeddings=embs)
    print(f"✅ Cached {len(ids):,} unique docs")
else:
    print(f"✅ Re-using existing cache with {coll.count():,} docs")

# ─────────────────────────────────────────────────────────────────────────────
# 3️⃣  Iterate queries, gather metrics
# ─────────────────────────────────────────────────────────────────────────────
print("🔎 Running evaluation …")
tot_loss, tot_f1, n_queries = 0.0, 0.0, len(test_ds)

for item in tqdm(test_ds):
    q_vec   = item["query"].unsqueeze(0).to(DEVICE)       # (1, dim)
    pos_vec = item["positive"].unsqueeze(0).to(DEVICE)
    neg_vec = item["negative"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        q_enc  = qry_enc(q_vec).cpu()                     # (1, dim)
        p_enc  = doc_enc(pos_vec).cpu()
        n_enc  = doc_enc(neg_vec).cpu()

    # ── Triplet loss
    loss_val = triplet_margin_loss(q_enc, p_enc, n_enc).item()
    tot_loss += loss_val

    # ── Retrieve top-k via Chroma
    results = coll.query(
        query_embeddings=q_enc.tolist(),  # we supply the embedding, not text
        n_results=K,
        include=["ids"]
    )
    retrieved_ids = results["ids"][0]    # list of ids

    # Build relevance vector (1 relevant doc per sample)
    relevant_id = None
    # find which cached id corresponds to the positive encoding
    # we stored raw tensor bytes as key originally, get first match:
    pos_key = item["positive"].cpu().numpy().tobytes()
    # simple O(N) search (tiny) — could store a reverse-lookup dict
    for id_, vec in coll.get(ids=retrieved_ids, include=["embeddings"])["embeddings"][0]:
        pass  # placeholder (embedding retrieval not needed here)

    # We stored mapping in 'id2vec'
    for _id, vec in id2vec.items():
        if torch.allclose(vec, doc_enc(item["positive"].to(DEVICE)).cpu(), atol=1e-6):
            relevant_id = _id
            break

    hits = 1 if relevant_id in retrieved_ids else 0
    precision = hits / K
    recall    = hits / 1        # exactly 1 positive per query
    f1        = 0.0 if hits == 0 else 2 * precision * recall / (precision + recall)

    tot_f1 += f1

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

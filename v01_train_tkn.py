#!/usr/bin/env python
"""
01_train_tkn.py
---------------
Tokenise the triplets produced by 00_build_triplets.py, build the
vocabulary, convert to ID sequences, and save everything.

This file **no longer touches the raw MS‑MARCO split** – it works
only with the JSONL triplets file.
"""
from __future__ import annotations
import os, sys, re, pickle, argparse, collections, json
from pathlib import Path

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus   import stopwords
from dotenv import load_dotenv
import wandb

# ────────────────────────────────────────────────────────────────────────────────
# 0️⃣  Config & CLI
# ────────────────────────────────────────────────────────────────────────────────
PAD_TOKEN    = "<PAD>"
TOKEN_RE     = re.compile(r"[^\w\s-]")
STOP_WORDS   = set()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--triplets", required=True,
                   help="JSONL produced by 00_build_triplets.py")
    p.add_argument("--top_n_words", type=int, default=50_000)
    p.add_argument("--save_dir",    default="data/tokens")
    p.add_argument("--wandb_project", default="ms‑marco‑tokeniser")
    return p.parse_args()

args = parse_args()
SAVE_DIR = Path(args.save_dir)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────────
# 1️⃣  NLTK setup
# ────────────────────────────────────────────────────────────────────────────────
for resource in ("punkt", "stopwords"):
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource=="punkt" else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)
STOP_WORDS = set(stopwords.words("english"))

def preprocess(text: str) -> list[str]:
    text = TOKEN_RE.sub(" ", text.lower()).replace("-", " ")
    return [w for w in word_tokenize(text) if w.isalpha() and w not in STOP_WORDS]

# ────────────────────────────────────────────────────────────────────────────────
# 2️⃣  Load triplets & tokenise
# ────────────────────────────────────────────────────────────────────────────────
print(f"📖  reading {args.triplets} …")
query_toks, pos_toks, neg_toks = [], [], []
all_tokens                     = []

with open(args.triplets, encoding="utf‑8") as f:
    for line in f:
        j = json.loads(line)
        q_t = preprocess(j["query"]);      query_toks.append(q_t); all_tokens.extend(q_t)
        p_t = preprocess(j["positive"]);   pos_toks.append(p_t);   all_tokens.extend(p_t)
        n_t = preprocess(j["negative"]);   neg_toks.append(n_t);   all_tokens.extend(n_t)

print(f"✅  tokenised {len(query_toks):,} triplets")

# ────────────────────────────────────────────────────────────────────────────────
# 3️⃣  Build vocabulary
# ────────────────────────────────────────────────────────────────────────────────
freq          = collections.Counter(all_tokens)
most_common   = [w for w, _ in freq.most_common(args.top_n_words)]
idx_to_word   = [PAD_TOKEN] + most_common
word_to_idx   = {w: i for i, w in enumerate(idx_to_word)}
print(f"🧠  vocab size (+PAD): {len(idx_to_word):,}")

def to_ids(toks: list[list[str]]) -> list[list[int]]:
    return [[word_to_idx.get(w, 0) for w in seq] for seq in toks]

q_ids   = to_ids(query_toks)
p_ids   = to_ids(pos_toks)
n_ids   = to_ids(neg_toks)
corpus  = [word_to_idx.get(w, 0) for w in all_tokens]

# ────────────────────────────────────────────────────────────────────────────────
# 4️⃣  Save artifacts
# ────────────────────────────────────────────────────────────────────────────────
out = {
    "query_tokens.pkl":        query_toks,
    "positive_tokens.pkl":     pos_toks,
    "negative_tokens.pkl":     neg_toks,
    "query_ids.pkl":           q_ids,
    "positive_ids.pkl":        p_ids,
    "negative_ids.pkl":        n_ids,
    "word_to_idx.pkl":         word_to_idx,
    "idx_to_word.pkl":         idx_to_word,
    "corpus_ids.pkl":          corpus,
}
for fname, obj in out.items():
    with open(SAVE_DIR / fname, "wb") as f:
        pickle.dump(obj, f)
print(f"💾  saved {len(out)} files ➜ {SAVE_DIR}")

# ────────────────────────────────────────────────────────────────────────────────
# 5️⃣  (optional) wandb logging
# ────────────────────────────────────────────────────────────────────────────────
load_dotenv()
if os.environ.get("WANDB_API_KEY"):
    run = wandb.init(project=args.wandb_project,
                     config=dict(top_n_words=args.top_n_words,
                                 num_triplets=len(query_toks)))
    run.log({"vocab_size": len(idx_to_word)})
    run.finish()

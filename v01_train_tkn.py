#!/usr/bin/env python
"""
01_train_tkn.py (enhanced)
-------------------------
Tokenise the triplets produced by 00_build_triplets.py, build the
vocabulary, convert to ID sequences, and save everything.

This script **never touches the raw MS‑MARCO split** – it works only with
 the JSONL triplets file.
"""
from __future__ import annotations
import os, sys, re, pickle, argparse, collections, json, datetime
from pathlib import Path
from typing import List, Dict

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus   import stopwords
from dotenv import load_dotenv

# Install wandb / huggingface‑hub on the fly if they are missing (handy on bare VMs)
try:
    import wandb  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – runtime convenience
    os.system(f"{sys.executable} -m pip install -q wandb")
    import wandb  # type: ignore

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
    p.add_argument("--save_dir",    default="data/tokens",
                   help="Where to write the pickle artefacts")
    p.add_argument("--wandb_project", default="ms‑marco‑tokeniser")
    p.add_argument("--wandb_run_name", default=None,
                   help="If set, overrides the automatic run name")
    p.add_argument("--hf_repo", default=None,
                   help="Hugging Face Hub repo name (e.g. user_name/dataset_name)."
                        " If omitted, no upload is attempted.")
    p.add_argument("--hf_repo_type", default="dataset", choices=["dataset", "model"],
                   help="Repo type on the Hub – usually 'dataset'.")
    return p.parse_args()


args = parse_args()
SAVE_DIR = Path(args.save_dir)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────────
# 1️⃣  NLTK setup
# ────────────────────────────────────────────────────────────────────────────────
for resource in ("punkt", "stopwords"):
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)
STOP_WORDS = set(stopwords.words("english"))


def preprocess(text: str) -> List[str]:
    """Lower‑case, strip punctuation / hyphens, tokenise, and drop stop words."""
    text = TOKEN_RE.sub(" ", text.lower()).replace("-", " ")
    return [w for w in word_tokenize(text) if w.isalpha() and w not in STOP_WORDS]


# ────────────────────────────────────────────────────────────────────────────────
# 2️⃣  Load triplets & tokenise
# ────────────────────────────────────────────────────────────────────────────────
print(f"📖  reading {args.triplets} …")
query_toks, pos_toks, neg_toks = [], [], []  # type: List[List[str]]
all_tokens: List[str] = []

with open(args.triplets, encoding="utf‑8") as f:
    for line in f:
        j = json.loads(line)
        q_t = preprocess(j["query"]);    query_toks.append(q_t); all_tokens.extend(q_t)
        p_t = preprocess(j["positive"]); pos_toks.append(p_t);   all_tokens.extend(p_t)
        n_t = preprocess(j["negative"]); neg_toks.append(n_t);   all_tokens.extend(n_t)

print(f"✅  tokenised {len(query_toks):,} triplets")

# ────────────────────────────────────────────────────────────────────────────────
# 3️⃣  Build vocabulary
# ────────────────────────────────────────────────────────────────────────────────
freq        = collections.Counter(all_tokens)
most_common = [w for w, _ in freq.most_common(args.top_n_words)]
idx_to_word = [PAD_TOKEN] + most_common
word_to_idx = {w: i for i, w in enumerate(idx_to_word)}
print(f"🧠  vocab size (+PAD): {len(idx_to_word):,}")


def to_ids(toks: List[List[str]]) -> List[List[int]]:
    return [[word_to_idx.get(w, 0) for w in seq] for seq in toks]


q_ids  = to_ids(query_toks)
p_ids  = to_ids(pos_toks)
n_ids  = to_ids(neg_toks)
corpus = [word_to_idx.get(w, 0) for w in all_tokens]

# ────────────────────────────────────────────────────────────────────────────────
# 4️⃣  Save artefacts locally
# ────────────────────────────────────────────────────────────────────────────────
print("💾  saving pickle artefacts …")
out: Dict[str, object] = {
    "query_tokens.pkl":    query_toks,
    "positive_tokens.pkl": pos_toks,
    "negative_tokens.pkl": neg_toks,
    "query_ids.pkl":       q_ids,
    "positive_ids.pkl":    p_ids,
    "negative_ids.pkl":    n_ids,
    "word_to_idx.pkl":     word_to_idx,
    "idx_to_word.pkl":     idx_to_word,
    "corpus_ids.pkl":      corpus,
}
for fname, obj in out.items():
    with open(SAVE_DIR / fname, "wb") as f:
        pickle.dump(obj, f)
print(f"✅  saved {len(out)} files ➜ {SAVE_DIR}\n")

# ────────────────────────────────────────────────────────────────────────────────
# 5️⃣  wandb logging
# ────────────────────────────────────────────────────────────────────────────────
print("📊  Initialising wandb …")
load_dotenv()
# Accept both WANDB_API_KEY (official) and legacy WANDDB_KEY env names
wandb_key = os.environ.get("WANDB_API_KEY") or os.environ.get("WANDDB_KEY")
if wandb_key:
    os.environ["WANDB_API_KEY"] = wandb_key  # ensure wandb sees it
else:
    print("⚠️  WANDB_API_KEY not found – skipping wandb logging.\n")

run = None
if os.environ.get("WANDB_API_KEY"):
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"tokenise-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=dict(
            triplets_file=args.triplets,
            top_n_words=args.top_n_words,
            num_triplets=len(query_toks),
            vocab_size=len(idx_to_word),
        ),
    )

    # Log simple scalars
    run.log({
        "vocab_size": len(idx_to_word),
        "num_triplets": len(query_toks),
    })

    # Attach all pickle files as a dataset‑type Artifact
    art = wandb.Artifact("tokeniser-output", type="dataset")
    for fname in out.keys():
        art.add_file(str(SAVE_DIR / fname))
    run.log_artifact(art)

    run.finish()
    print("✅  wandb run finished and artefacts logged.\n")

# ────────────────────────────────────────────────────────────────────────────────
# 6️⃣  Upload to Hugging Face Hub
# ────────────────────────────────────────────────────────────────────────────────
if args.hf_repo:
    print("🚀  Uploading artefacts to the Hugging Face Hub …")
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:  # pragma: no cover – runtime convenience
        print("📦  Installing huggingface‑hub …")
        os.system(f"{sys.executable} -m pip install -q huggingface_hub")
        from huggingface_hub import HfApi, create_repo

    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HUGGINGFACE_KEY")
    if not hf_token:
        print("⚠️  HUGGINGFACE_TOKEN not found – cannot push to the Hub. Skipping.\n")
    else:
        api = HfApi()
        # Create repo if it doesn't yet exist
        create_repo(repo_id=args.hf_repo, repo_type=args.hf_repo_type, token=hf_token, exist_ok=True)

        # Craft README / model card
        readme = f"""---
license: mit
tags:
- ms‑marco
- tokeniser
---

# MS‑MARCO Tokeniser Output

This repository stores vocabulary & ID mappings produced by `01_train_tkn.py`.

* **Triplets file**: `{args.triplets}`
* **Top‑N words kept**: {args.top_n_words}
* **Vocab size (incl. PAD)**: {len(idx_to_word):,}
* **Triplets processed**: {len(query_toks):,}

The pickle files can be loaded via `pickle.load(open(fname, 'rb'))`.
"""
        Path(SAVE_DIR / "README.md").write_text(readme)

        # Which files to push?  All pickle files + README.
        files_to_upload = [SAVE_DIR / f for f in out.keys()] + [SAVE_DIR / "README.md"]

        for fp in files_to_upload:
            print(f"  ↳ uploading {fp.name}")
            api.upload_file(
                path_or_fileobj=str(fp),
                path_in_repo=fp.name,
                repo_id=args.hf_repo,
                repo_type=args.hf_repo_type,
                token=hf_token,
            )
        print(f"✅  Artefacts available at https://huggingface.co/{args.hf_repo}\n")
else:
    print("ℹ️  --hf_repo not provided; skipping Hugging Face upload.")

print("🎉  Done!")

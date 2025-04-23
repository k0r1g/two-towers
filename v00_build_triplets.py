#!/usr/bin/env python
"""
00_build_triplets.py
--------------------
Create (query, positive, negative) triplets from MS‑MARCO and
save them as JSONL – one triplet per line.

▶︎ Usage
$ python 00_build_triplets.py \
      --version v1.1 \
      --num_examples 5000 \
      --split train \
      --outfile data/marco_triplets_5k.jsonl
"""
from dotenv import load_dotenv
load_dotenv() 

import random, json, argparse
from pathlib import Path
from datasets import load_dataset
from tqdm.auto import tqdm

def build_triplets(version: str,
                   split: str,
                   num_examples: int | None,
                   outfile: str) -> Path:
    ds = load_dataset("microsoft/ms_marco", version, split=split)
    if num_examples:
        ds = ds.shuffle(seed=42).select(range(num_examples))

    path = Path(outfile)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf‑8") as f:
        for idx in tqdm(range(len(ds)), desc="triplets"):
            row       = ds[idx]
            query     = row["query"]
            passages  = row["passages"]["passage_text"]
            selected  = row["passages"]["is_selected"]

            # -------- relevant passage --------
            try:
                rel_idx = selected.index(1)
            except ValueError:          # no pos passage – skip
                continue
            positive = passages[rel_idx]

            # -------- random negative --------
            while True:
                j = random.randrange(len(ds))
                if j == idx:                           # must be OTHER query
                    continue
                neg_passages = ds[j]["passages"]["passage_text"]
                if neg_passages:
                    negative = random.choice(neg_passages)
                    break

            json.dump({"query": query,
                       "positive": positive,
                       "negative": negative}, f, ensure_ascii=False)
            f.write("\n")

    print(f"📝  saved {sum(1 for _ in path.open()):,} triplets ➜ {path}")
    return path

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="v1.1")
    p.add_argument("--split",   default="train")
    p.add_argument("--num_examples", type=int)
    p.add_argument("--outfile", default="data/marco_triplets.jsonl")
    build_triplets(**vars(p.parse_args()))

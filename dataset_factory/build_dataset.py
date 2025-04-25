#!/usr/bin/env python
"""
Example:
$ python -m dataset_factory.build_dataset \
    --preset presets/multi_pos_multi_neg.yml \
    --split train \
    --output data/processed/multi_pos_multi_neg.parquet
"""
import argparse, yaml, random
import pandas as pd
from pathlib import Path

from .readers import load_split
from .positive_selectors import classic_positives
from .negative_samplers import build_inter_query_pool, random_inter_query, intra_query_zero_filtered

SELECTORS = {
    "classic": classic_positives,
}

SAMPLERS = {
    "random_inter": random_inter_query,
    "intra_zero_filtered": intra_query_zero_filtered,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--output", required=True)
    ap.add_argument("--neg_k", type=int, default=1,
                    help="negatives per positive")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.preset).read_text())

    print(f"Loading split {args.split} …")
    df = load_split(args.split)

    # Positive generator
    selector_fn = SELECTORS[cfg["positive_selector"]]
    positives = list(selector_fn(df))

    # Negative sampler
    if cfg["negative_sampler"]["type"] == "random_inter":
        pool = build_inter_query_pool(df)
        neg_sampler = random_inter_query(pool)
    elif cfg["negative_sampler"]["type"] == "intra_query_zero":
        neg_sampler = intra_query_zero_filtered()
    else:
        raise ValueError("unknown negative_sampler")

    rows_out = []

    for pos in positives:
        q = pos["query"]
        d_pos = pos["positive"]
        for _ in range(cfg.get("negatives_per_pos", 1)):
            d_neg = None
            trials = 0
            while d_neg is None and trials < 5:
                d_neg = neg_sampler({**pos, "row": df.iloc[random.randrange(len(df))]})
                trials += 1
            if d_neg is None:
                continue   # fall back
            rows_out.append((q, d_pos, d_neg))

    out_df = pd.DataFrame(rows_out, columns=["q_text", "d_pos_text", "d_neg_text"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output)
    print(f"Wrote {len(out_df):,} triplets to {args.output}")

if __name__ == "__main__":
    main() 
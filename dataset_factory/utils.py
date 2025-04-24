import re
from typing import List, Set

def flatten_answers(row) -> List[str]:
    return row["answers"] or []   # handles empty lists (HF sets None)

def answer_in_text(text: str, answers: List[str]) -> bool:
    for a in answers:
        # crude but fast: case-insensitive substring
        if a and a.lower() in text.lower():
            return True
    return False

def ngram_set(text: str, n: int = 3) -> Set[str]:
    tokens = text.lower().split()
    return {" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)}

def high_ngram_overlap(p1: str, p2: str, thresh: float = 0.8) -> bool:
    ngrams1, ngrams2 = ngram_set(p1), ngram_set(p2)
    if not ngrams1 or not ngrams2:
        return False
    jacc = len(ngrams1 & ngrams2) / len(ngrams1 | ngrams2)
    return jacc >= thresh 
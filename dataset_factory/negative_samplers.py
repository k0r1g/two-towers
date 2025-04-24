import random
from typing import List, Dict, Callable
from .utils import answer_in_text, high_ngram_overlap

def build_inter_query_pool(df) -> List[str]:
    pool = []
    for _, row in df.iterrows():
        passages = row["passages"]
        passage_texts = passages["passage_text"]
        for text in passage_texts:
            pool.append(text)
    return pool

def random_inter_query(pool: List[str]) -> Callable[[Dict], str]:
    def _sample(_: Dict) -> str:
        return random.choice(pool)
    return _sample

def intra_query_zero_filtered() -> Callable[[Dict], str]:
    """
    Assumes the caller passes the *original MS-MARCO row* in the dict.
    Filters: no answer string, low n-gram overlap with positive.
    """
    def _sample(ctx: Dict) -> str:
        pos_text = ctx["positive"]
        answers = ctx["answers"]
        
        row = ctx["row"]
        passages = row["passages"]
        is_selected = passages["is_selected"]
        passage_texts = passages["passage_text"]
        
        zero_passages = []
        for i, selected in enumerate(is_selected):
            if selected == 0:
                text = passage_texts[i]
                if not answer_in_text(text, answers) and not high_ngram_overlap(text, pos_text):
                    zero_passages.append(text)
        
        return random.choice(zero_passages) if zero_passages else None
    return _sample 
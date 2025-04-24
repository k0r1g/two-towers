from typing import Iterable, Dict
import numpy as np

def classic_positives(df) -> Iterable[Dict]:
    """
    Yield dicts: {'query': str, 'positive': str, 'answers': List[str]}
    One dict per is_selected==1 passage (multi-positive ready).
    """
    for _, row in df.iterrows():
        passages = row["passages"]
        is_selected = passages["is_selected"]
        passage_texts = passages["passage_text"]
        
        for i, selected in enumerate(is_selected):
            if selected == 1:
                yield {
                    "query": row["query"],
                    "positive": passage_texts[i],
                    "answers": row["answers"] if isinstance(row["answers"], list) else [],
                } 
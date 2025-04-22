# dataset.py
# ----------
import pickle
import torch
from torch.utils.data import Dataset

class MSMarcoCBOWDataset(Dataset):
    def __init__(self, window=4, data_dir="data/train_tokens"):
        self.window = window
        self.data_dir = data_dir

        with open(f"{data_dir}/corpus_ids.pkl", "rb") as f:
            self.corpus_ids = pickle.load(f)

        with open(f"{data_dir}/word_to_idx.pkl", "rb") as f:
            self.word_to_idx = pickle.load(f)

        self.vocab_size = len(self.word_to_idx)
        self.pairs = self._generate_pairs()

    def _generate_pairs(self):
        pairs = []
        for i in range(self.window, len(self.corpus_ids) - self.window):
            context = (
                self.corpus_ids[i - self.window : i] +
                self.corpus_ids[i + 1 : i + 1 + self.window]
            )
            target = self.corpus_ids[i]
            if len(context) == 2 * self.window:
                pairs.append((context, target))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        context, target = self.pairs[idx]
        return torch.tensor(context), torch.tensor(target)





class TripletDataset(Dataset):
    def __init__(self, triplets, tokenizer, embedder):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.embedder = embedder

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        q, p, n = self.triplets[idx]
        return {
            'query': self.embedder(self.tokenizer(q)),
            'positive': self.embedder(self.tokenizer(p)),
            'negative': self.embedder(self.tokenizer(n)),
        }
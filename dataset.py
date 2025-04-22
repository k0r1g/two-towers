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
    def __init__(self, data_dir="data/train_tokens"):
        self.data_dir = data_dir

        with open(f"{data_dir}/query_ids.pkl", "rb") as f:
            self.query_ids = pickle.load(f)

        with open(f"{data_dir}/positive_ids.pkl", "rb") as f:
            self.positive_ids = pickle.load(f)

        with open(f"{data_dir}/negative_ids.pkl", "rb") as f:
            self.negative_ids = pickle.load(f)

        assert len(self.query_ids) == len(self.positive_ids) == len(self.negative_ids)

    def __len__(self):
        return len(self.query_ids)

    def __getitem__(self, idx):
        return {
            "query": torch.tensor(self.query_ids[idx], dtype=torch.long),
            "positive": torch.tensor(self.positive_ids[idx], dtype=torch.long),
            "negative": torch.tensor(self.negative_ids[idx], dtype=torch.long),
        }

class DualEncoderDataset(Dataset):
    """Dataset for Dual Encoder model using precomputed averaged embeddings."""
    def __init__(self, triplets):
        # triplets: List of tuples, where each tuple is (query_embedding, positive_embedding, negative_embedding)
        self.data = triplets

    def __len__(self):
        # Returns the total number of triplets
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieves the triplet at the given index
        q, p, n = self.data[idx]
        # Returns a dictionary containing the query, positive, and negative embeddings
        return {'query': q, 'positive': p, 'negative': n}
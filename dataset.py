# dataset.py
# ----------
import torch
import pickle


class MSMarcoCBOWDataset(torch.utils.data.Dataset):
    def __init__(self, window=4):
        self.word_to_idx = pickle.load(open("word_to_idx.pkl", "rb"))
        self.query_tokens = pickle.load(open("query_token_ids.pkl", "rb"))
        self.rel_pass_tokens = pickle.load(open("relevant_token_ids.pkl", "rb"))
        self.irrel_pass_tokens = pickle.load(open("irrelevant_token_ids.pkl", "rb"))

        self.window = window
        self.pairs = []
        self.build_pairs()

    def build_pairs(self):
        all_sequences = self.query_tokens + self.rel_pass_tokens + self.irrel_pass_tokens
        for tokens in all_sequences:
            if len(tokens) < 2 * self.window + 1:
                continue
            for i in range(self.window, len(tokens) - self.window):
                context = tokens[i - self.window:i] + tokens[i + 1:i + self.window + 1]
                target = tokens[i]
                self.pairs.append((context, target))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        context, target = self.pairs[idx]
        return torch.tensor(context), torch.tensor(target)

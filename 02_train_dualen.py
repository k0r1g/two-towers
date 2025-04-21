# 02_train_dualen.py – train two-tower model on MS MARCO triplets
# ---------------------------------------------------------------
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 1️⃣  Parameters
# ---------------------------------------------------------------------------
BATCH_SIZE   = 32
EMB_DIM      = 100  # matches the Word2Vec dimension
LR           = 1e-3
EPOCHS       = 5
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR     = Path(".")  # or wherever your .pkl files live
MARGIN       = 0.2

# ---------------------------------------------------------------------------
# 2️⃣  Load tokenised data
# ---------------------------------------------------------------------------
print("📦 Loading tokenised data …")
query_token_ids          = pickle.load(open(DATA_DIR / "query_token_ids.pkl", "rb"))
relevant_token_ids       = pickle.load(open(DATA_DIR / "relevant_token_ids.pkl", "rb"))
irrelevant_token_ids     = pickle.load(open(DATA_DIR / "irrelevant_token_ids.pkl", "rb"))
word_to_idx              = pickle.load(open(DATA_DIR / "word_to_idx.pkl", "rb"))
idx_to_word              = pickle.load(open(DATA_DIR / "idx_to_word.pkl", "rb"))

vocab_size = len(idx_to_word)

# ---------------------------------------------------------------------------
# 3️⃣  Dummy Word2Vec Embedding Layer (replace with pretrained)
# ---------------------------------------------------------------------------
embedding_layer = nn.Embedding(vocab_size, EMB_DIM)
embedding_layer.weight.data.normal_(mean=0, std=0.1)
embedding_layer = embedding_layer.to(DEVICE)

# Optional: freeze embeddings
embedding_layer.weight.requires_grad = False

# ---------------------------------------------------------------------------
# 4️⃣  Dataset + Triplet Sampling
# ---------------------------------------------------------------------------
class MarcoTripletDataset(Dataset):
    def __init__(self, queries, relevant_docs, irrelevant_docs):
        self.queries = queries
        self.relevant_docs = relevant_docs
        self.irrelevant_docs = irrelevant_docs

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        q = self.queries[idx]

        # Randomly sample 1 relevant and 1 irrelevant doc for this query
        pos_doc = random.choice(self.relevant_docs)
        neg_doc = random.choice(self.irrelevant_docs)

        return {
            "query": torch.tensor(q, dtype=torch.long),
            "pos": torch.tensor(pos_doc, dtype=torch.long),
            "neg": torch.tensor(neg_doc, dtype=torch.long),
        }

def collate_fn(batch):
    def avg_embed(sequences):
        masks = (sequences != 0).unsqueeze(-1).float()
        embeds = embedding_layer(sequences)
        summed = torch.sum(embeds * masks, dim=1)
        lengths = masks.sum(dim=1).clamp(min=1e-9)
        return summed / lengths

    q_batch = torch.nn.utils.rnn.pad_sequence([b["query"] for b in batch], batch_first=True).to(DEVICE)
    p_batch = torch.nn.utils.rnn.pad_sequence([b["pos"] for b in batch], batch_first=True).to(DEVICE)
    n_batch = torch.nn.utils.rnn.pad_sequence([b["neg"] for b in batch], batch_first=True).to(DEVICE)

    q_vec = avg_embed(q_batch)
    p_vec = avg_embed(p_batch)
    n_vec = avg_embed(n_batch)

    return q_vec, p_vec, n_vec

triplet_dataset = MarcoTripletDataset(query_token_ids, relevant_token_ids, irrelevant_token_ids)
triplet_loader = DataLoader(triplet_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# ---------------------------------------------------------------------------
# 5️⃣  Model definition
# ---------------------------------------------------------------------------
class QryTower(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc(x)

class DocTower(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc(x)

qryTower = QryTower(EMB_DIM).to(DEVICE)
docTower = DocTower(EMB_DIM).to(DEVICE)

# ---------------------------------------------------------------------------
# 6️⃣  Optimizer + Training loop
# ---------------------------------------------------------------------------
params = list(qryTower.parameters()) + list(docTower.parameters())
optimizer = torch.optim.Adam(params, lr=LR)

print("🚀 Starting training …")
for epoch in range(EPOCHS):
    qryTower.train()
    docTower.train()
    epoch_loss = 0.0

    for qry_vecs, pos_vecs, neg_vecs in tqdm(triplet_loader):
        q = qryTower(qry_vecs)
        pos = docTower(pos_vecs)
        neg = docTower(neg_vecs)

        sim_pos = F.cosine_similarity(q, pos)
        sim_neg = F.cosine_similarity(q, neg)

        triplet_loss = torch.clamp(MARGIN - (sim_pos - sim_neg), min=0.0).mean()

        optimizer.zero_grad()
        triplet_loss.backward()
        optimizer.step()

        epoch_loss += triplet_loss.item()

    avg_loss = epoch_loss / len(triplet_loader)
    print(f"🧪 Epoch {epoch+1}/{EPOCHS} – Loss: {avg_loss:.4f}")

# ---------------------------------------------------------------------------
# 7️⃣  Save model
# ---------------------------------------------------------------------------
print("💾 Saving model weights …")
torch.save(qryTower.state_dict(), "qry_tower.pt")
torch.save(docTower.state_dict(), "doc_tower.pt")
print("✅ Training complete.")

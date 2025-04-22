#!/usr/bin/env python
"""
04_train_dualen.py
-------------------
Trains a two-tower dual encoder model using preprocessed triplet datasets.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 1️⃣  Config
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
EMB_DIM    = 128      # must match Word2Vec dim
LR         = 1e-3
EPOCHS     = 5
MARGIN     = 0.2
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR   = Path("data/dualen")
SAVE_DIR   = Path("checkpoints/dualen")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2️⃣  Load datasets
# ─────────────────────────────────────────────────────────────────────────────
print("📦 Loading datasets …")
train_ds = torch.load(DATA_DIR / "train_dualen.pt")
val_ds   = torch.load(DATA_DIR / "val_dualen.pt")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

print(f"✅ Loaded {len(train_ds):,} train samples | {len(val_ds):,} val samples")

# ─────────────────────────────────────────────────────────────────────────────
# 3️⃣  Dual Encoder Towers
# ─────────────────────────────────────────────────────────────────────────────
class Tower(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)  # Optional second layer
        )

    def forward(self, x):
        return self.fc(x)

query_encoder = Tower(EMB_DIM).to(DEVICE)
doc_encoder   = Tower(EMB_DIM).to(DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
# 4️⃣  Optimizer + Loss
# ─────────────────────────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(
    list(query_encoder.parameters()) + list(doc_encoder.parameters()),
    lr=LR
)

def triplet_loss(q, pos, neg, margin=MARGIN):
    sim_pos = F.cosine_similarity(q, pos)
    sim_neg = F.cosine_similarity(q, neg)
    return torch.clamp(margin - (sim_pos - sim_neg), min=0.0).mean()

# ─────────────────────────────────────────────────────────────────────────────
# 5️⃣  Training Loop
# ─────────────────────────────────────────────────────────────────────────────
print("🚀 Starting training …")
for epoch in range(EPOCHS):
    query_encoder.train()
    doc_encoder.train()
    train_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        q = batch['query'].to(DEVICE)
        p = batch['positive'].to(DEVICE)
        n = batch['negative'].to(DEVICE)

        q_vec = query_encoder(q)
        p_vec = doc_encoder(p)
        n_vec = doc_encoder(n)

        loss = triplet_loss(q_vec, p_vec, n_vec)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    print(f"📉 Epoch {epoch+1} train loss: {avg_train_loss:.4f}")

    # Optional: evaluate on val
    query_encoder.eval()
    doc_encoder.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            q = batch['query'].to(DEVICE)
            p = batch['positive'].to(DEVICE)
            n = batch['negative'].to(DEVICE)

            q_vec = query_encoder(q)
            p_vec = doc_encoder(p)
            n_vec = doc_encoder(n)

            loss = triplet_loss(q_vec, p_vec, n_vec)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"🧪 Epoch {epoch+1} val loss: {avg_val_loss:.4f}")

    # Save checkpoint
    torch.save({
        'query_encoder': query_encoder.state_dict(),
        'doc_encoder': doc_encoder.state_dict(),
        'epoch': epoch + 1,
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss
    }, SAVE_DIR / f"dualen_epoch{epoch+1}.pt")

print("✅ Training complete.")

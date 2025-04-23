#!/usr/bin/env python
"""
v04_train_dualen.py
-------------------
Trains a two-tower dual encoder model using preprocessed triplet datasets.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# Import the tower models
from model import QryTower, DocTower

# ── Extra integrations ─────────────────────────────────────────────────────
import datetime, os, sys
from dotenv import load_dotenv
import wandb

# Hugging Face (auto-install if missing)
try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("📦 Installing huggingface_hub …")
    os.system(f"{sys.executable} -m pip install -q huggingface_hub")
    from huggingface_hub import HfApi, create_repo

# ─────────────────────────────────────────────────────────────────────────────
# 1️⃣  Config
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
EMB_DIM    = 128      # must match embedding dim used in data prep
LR         = 1e-3
EPOCHS     = 5
MARGIN     = 0.2
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR   = Path("data/dualen")
SAVE_DIR   = Path("checkpoints/dualen")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── wandb & HF config ───────────────────────────────────────────────────────
load_dotenv()

WANDB_PROJECT = "msmarco-dualen-train"
RUN_NAME      = f"dualen-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

HF_REPO_ID   = "Kogero/msmarco-dualen-mlp"
HF_REPO_TYPE = "model"

# ─────────────────────────────────────────────────────────────────────────────
# ── Initialize wandb run ────────────────────────────────────────────────────
run = wandb.init(project=WANDB_PROJECT, name=RUN_NAME, config={
    "batch_size": BATCH_SIZE,
    "embed_dim":  EMB_DIM,
    "lr":         LR,
    "margin":     MARGIN,
    "epochs":     EPOCHS,
})

# ─────────────────────────────────────────────────────────────────────────────
# 2️⃣  Load datasets
# ─────────────────────────────────────────────────────────────────────────────
print("📦 Loading datasets …")
#(If you decide to train an RNN version later, just swap in *_dualen_seq.pt.)
train_ds = torch.load(DATA_DIR / "train_dualen_avg.pt")   # pooled for MLP towers
val_ds   = torch.load(DATA_DIR / "val_dualen_avg.pt")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

print(f"✅ Loaded {len(train_ds):,} train samples | {len(val_ds):,} val samples")

# ─────────────────────────────────────────────────────────────────────────────
# 3️⃣  Dual Encoder Towers
# ─────────────────────────────────────────────────────────────────────────────
# We now instantiate the specific tower classes imported from model.py
query_encoder = QryTower(EMB_DIM).to(DEVICE)
doc_encoder   = DocTower(EMB_DIM).to(DEVICE)

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
#   NOTES:
# ─────────────────────────────────────────────────────────────────────────────  
'''
q_input ──▶ [query_encoder weights] ──▶ q_vec ─┐
                                               │
p_input ──▶ [doc_encoder weights]   ──▶ p_vec ─┼──▶ triplet_loss ─▶ loss
n_input ──▶ [doc_encoder weights]   ──▶ n_vec ─┘


q_input → query_encoder → loss

p_input → doc_encoder → loss

n_input → doc_encoder → loss



''' 
    

# ─────────────────────────────────────────────────────────────────────────────
# 5️⃣  Training Loop
# ─────────────────────────────────────────────────────────────────────────────
print("🚀 Starting training …")
for epoch in range(EPOCHS):
    query_encoder.train()
    doc_encoder.train()
    train_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        # Unpack tuple and move tensors to device
        q, p, n = (t.to(DEVICE) for t in batch)

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
            # Unpack tuple and move tensors to device
            q, p, n = (t.to(DEVICE) for t in batch)

            q_vec = query_encoder(q)
            p_vec = doc_encoder(p)
            n_vec = doc_encoder(n)

            loss = triplet_loss(q_vec, p_vec, n_vec)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"🧪 Epoch {epoch+1} val loss: {avg_val_loss:.4f}")

    # ── Log metrics to wandb
    run.log({
        "epoch":       epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss":   avg_val_loss,
    })

    # Save checkpoint
    ckpt_path = SAVE_DIR / f"dualen_epoch{epoch+1}.pt"
    torch.save({
        'query_encoder': query_encoder.state_dict(),
        'doc_encoder': doc_encoder.state_dict(),
        'epoch': epoch + 1,
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss
    }, ckpt_path)

    # Log checkpoint as artifact - MOVED OUTSIDE LOOP
    # art = wandb.Artifact(f"dualen-ckpt-epoch{epoch+1}", type="model")
    # art.add_file(ckpt_path)
    # run.log_artifact(art)

# ── Log final checkpoint as artifact ────────────────────────────────────────
print(f"🪵 Logging final checkpoint {ckpt_path} to wandb...")
art = wandb.Artifact(f"dualen-ckpt-final", type="model", metadata={"epoch": EPOCHS}) # Use a fixed name, add epoch metadata
art.add_file(ckpt_path)
run.log_artifact(art)

# ── Upload last checkpoint to Hugging Face Hub ──────────────────────────────
hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HUGGINGFACE_KEY")
if hf_token:
    last_ckpt = ckpt_path  # from final epoch
    api = HfApi()
    create_repo(repo_id=HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
                token=hf_token,
                exist_ok=True)

    # Simple model card
    (SAVE_DIR / "README.md").write_text(f"""---
license: mit
tags:
+- dual-encoder
+- ms-marco
+- mlp
+---
+# MS-MARCO Dual Encoder (MLP towers)
+
+Trained with margin={MARGIN}, epochs={EPOCHS}, embedding dim={EMB_DIM}.
+""")

    for local, remote in [(last_ckpt, "dualen_model.pt"), (SAVE_DIR / "README.md", "README.md")]:
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            token=hf_token
        )
    print(f"🚀 Model uploaded: https://huggingface.co/{HF_REPO_ID}")
else:
    print("⚠️  HUGGINGFACE_TOKEN not set – skipping HF upload.")

# ── Finalise wandb ─────────────────────────────────────────────────────────
run.finish()

print("✅ Training complete.")

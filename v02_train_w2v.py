# 02_train_w2v.py – Train CBOW Word2Vec model on MS MARCO
# --------------------------------------------------------
import tqdm
import torch
import datetime
import wandb
import os
import sys
from pathlib import Path
from torch.utils.data import DataLoader

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

from dataset import MSMarcoCBOWDataset
from model import CBOW

# ---------------------------------------------------------------------------
# 0️⃣  Parameters + Setup
# ---------------------------------------------------------------------------
EMBED_DIM  = 128
WINDOW     = 4
BATCH_SIZE = 256
EPOCHS     = 5
LR         = 0.003
ts         = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

torch.manual_seed(42)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create checkpoint directory if it doesn't exist
os.makedirs("./checkpoints", exist_ok=True)

# ---------------------------------------------------------------------------
# 1️⃣  Load dataset
# ---------------------------------------------------------------------------
ds = MSMarcoCBOWDataset(window=WINDOW)
dl = DataLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True)
vocab_size = len(ds.word_to_idx)
print(f"🧠 Training pairs: {len(ds):,}")
print(f"📚 Vocab size: {vocab_size:,}")

# ---------------------------------------------------------------------------
# 2️⃣  Init model + training setup
# ---------------------------------------------------------------------------
model = CBOW(vocab_size=vocab_size, embed_dim=EMBED_DIM).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = torch.nn.CrossEntropyLoss()

model_params = sum(p.numel() for p in model.parameters())
print("🦾 Model parameters:", model_params)

# Initialize wandb with enhanced logging
print("🔄 Initializing wandb tracking...")
# Get wandb API key from environment
wandb_key = os.environ.get("WANDDB_KEY")  # Using WANDDB_KEY as specified
if not wandb_key:
    print("Warning: WANDDB_KEY not found in .env file")
    wandb_key = input("Enter your Weights & Biases API key: ")
    os.environ["WANDB_API_KEY"] = wandb_key
else:
    # Set the environment variable that wandb actually uses
    os.environ["WANDB_API_KEY"] = wandb_key

# Start a new wandb run with detailed config
run = wandb.init(
    project="msmarco-w2v", 
    name=f"cbow-{ts}",
    config={
        "embedding_dim": EMBED_DIM,
        "window_size": WINDOW,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "device": str(dev),
        "model_parameters": model_params,
        "vocab_size": vocab_size,
        "training_pairs": len(ds)
    }
)
print("✅ Tracking initialized")

# ---------------------------------------------------------------------------
# 3️⃣  Training Loop
# ---------------------------------------------------------------------------
final_model_path = f"./checkpoints/cbow_{ts}_final.pth"

for epoch in range(EPOCHS):
    pbar = tqdm.tqdm(dl, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)
    model.train()
    epoch_loss = 0.0
    
    for i, (ctx, tgt) in enumerate(pbar):
        ctx, tgt = ctx.to(dev), tgt.to(dev)
        opt.zero_grad()
        out = model(ctx)
        loss = loss_fn(out, tgt)
        loss.backward()
        opt.step()
        
        # Track batch loss
        loss_val = loss.item()
        epoch_loss += loss_val
        
        # Update progress bar and log batch metrics
        pbar.set_postfix(loss=f"{loss_val:.4f}")
        run.log({
            "batch_loss": loss_val,
            "epoch": epoch + 1,
            "batch": i,
            "progress": (i / len(dl)) + epoch
        })
    
    # Log epoch metrics
    avg_epoch_loss = epoch_loss / len(dl)
    run.log({
        "epoch": epoch + 1,
        "epoch_loss": avg_epoch_loss,
        "epoch_progress": (epoch + 1) / EPOCHS * 100
    })
    
    print(f"Epoch {epoch+1}/{EPOCHS} completed - Avg Loss: {avg_epoch_loss:.6f}")

    # save checkpoint
    ckpt = f"./checkpoints/cbow_{ts}_epoch{epoch+1}.pth"
    torch.save(model.state_dict(), ckpt)
    
    # Log checkpoint as artifact
    art = wandb.Artifact(f"cbow-model-epoch{epoch+1}", type="model")
    art.add_file(ckpt)
    run.log_artifact(art)

# Save final model
torch.save(model.state_dict(), final_model_path)
print(f"💾 Final model saved to {final_model_path}")

# Log final metrics
run.log({
    "completed": True,
    "final_loss": avg_epoch_loss
})

# ---------------------------------------------------------------------------
# 4️⃣  Upload model to Hugging Face
# ---------------------------------------------------------------------------
print("\nUploading to Hugging Face Hub …")
try:
    from huggingface_hub import HfApi
except ImportError:
    print("Installing huggingface_hub...")
    os.system(f"{sys.executable} -m pip install -q huggingface_hub")
    from huggingface_hub import HfApi

# Get Hugging Face token from environment
hf_token = os.environ.get("HUGGINGFACE_KEY")  # Using HUGGINGFACE_KEY as specified
if not hf_token:
    print("Warning: HUGGINGFACE_KEY not found in .env file")
    hf_token = input("Enter your Hugging Face access token: ")

api = HfApi()
repo_id = "Kogero/ms-marco-word2vec"
try:
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=hf_token)
    
    # Save model card (README.md) with metadata
    model_card = f"""---
language: en
license: mit
datasets:
- microsoft/ms_marco
tags:
- word2vec
- cbow
- continuous bag of words
- embedding
---

# MS MARCO Word2Vec Embedding Model

This repository contains a Continuous Bag of Words (CBOW) Word2Vec model trained on the Microsoft MS MARCO dataset.

## Model Details

- **Architecture**: CBOW (Continuous Bag of Words)
- **Embedding Dimension**: {EMBED_DIM}
- **Context Window Size**: {WINDOW}
- **Vocabulary Size**: {vocab_size:,}
- **Training Pairs**: {len(ds):,}
- **Parameters**: {model_params:,}
- **Training Device**: {dev}

## Usage

```python
import torch

# Load the model
vocab_size = {vocab_size}
embed_dim = {EMBED_DIM}
model = CBOW(vocab_size=vocab_size, embed_dim=embed_dim)
model.load_state_dict(torch.load("cbow_model.pth"))

# Get embeddings for words
embeddings = model.embeddings.weight  # Shape: [vocab_size, embed_dim]
```

## Training

This model was trained for {EPOCHS} epochs with a batch size of {BATCH_SIZE} and learning rate of {LR}.

## License

MIT
"""
    
    with open("README.md", "w") as f:
        f.write(model_card)
    
    # Also save a simple vocab file
    with open("vocab.txt", "w") as f:
        for word, idx in sorted(ds.word_to_idx.items(), key=lambda x: x[1]):
            f.write(f"{word}\n")
    
    # Define files to upload
    files_to_upload = {
        final_model_path: "cbow_model.pth",
        "README.md": "README.md",
        "vocab.txt": "vocab.txt",
    }
    
    # Upload files
    for local_path, remote_path in files_to_upload.items():
        print(f"  ↳ uploading {local_path} → {remote_path}")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="model",
            token=hf_token
        )
    
    print(f"\n✅  Done – model at https://huggingface.co/{repo_id}")
    
except Exception as e:
    print(f"❌ Error uploading to Hugging Face: {e}")

# Finish wandb run at the very end
run.finish()
print("🎉 Training complete!")

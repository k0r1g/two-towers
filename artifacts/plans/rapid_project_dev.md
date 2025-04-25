## 🚀 11 : 30 → 15 : 00 — “Zero-to-Prod” Checklist  
*(all timestamps are **wall-clock goals**, adjust if a step finishes early)*  

> **Legend**  
> • **GPU-box** = Ubuntu 22.04 machine with NVIDIA GPU (where you already SSH’d)<br>  
> • **VPS**   = lightweight public server that will host `chromadb` and an inference API (no GPU)<br>  
> • **Laptop** = your local dev machine (Cursor, Git, etc.)

---

### 0. Pre-flight (Already done)  
SSH shells live:  

```bash
# GPU-box
ssh gpu-user@GPUTOWER

# VPS
ssh vps-user@SEMANTICSEARCH
```

Clone/update repo on both:

```bash
git clone https://github.com/<you>/two-tower.git   # or cd two-tower && git pull
```

---

## 1 ️⃣ 11 : 30 – 11 : 45 🖥️ Hardware sanity

| Host | Commands | Expected |
|------|----------|----------|
| **GPU-box** | `nvidia-smi` | GPU name + driver OK |
|  | `python - <<'PY'\nimport torch,os;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))\nPY` | `True`, GPU model |
| **VPS** | `lscpu | grep -E 'Model name|CPU\(s\)'` | basic CPU info |

---

## 2 ️⃣ 11 : 45 – 12 : 20 🛠️ Environment bootstrap

### 2.1 GPU-box (training)

```bash
# create env
conda create -y -n twotower python=3.11
conda activate twotower

# core deps
pip install torch==2.2.0+cu121 -f https://download.pytorch.org/whl/cu121/torch_stable.html
pip install -r requirements.txt  # contains chromadb, wandb, pyyaml, etc.

# editable install
pip install -e .
wandb login   # paste API key once
```

### 2.2 VPS (serving)

```bash
sudo apt update && sudo apt install -y docker.io docker-compose
```

Create `docker-compose.yml` on VPS (will run Chroma + tiny FastAPI):

```yaml
version: "3.9"
services:
  chroma:
    image: chromadb/chroma:latest
    volumes:
      - ./chroma_data:/chroma/.chroma
    environment:
      - IS_PERSISTENT=TRUE
    ports: ["8000:8000"]

  inference:
    build: ./docker/inference
    environment:
      - MODEL_REPO_URL=https://huggingface.co/azuremis/twotower-char-emb
      - CHROMA_HOST=chroma
    depends_on: [chroma]
    ports: ["8080:8080"]
```

*(you’ll create `docker/inference/Dockerfile` at **14 : 10** step)*

```bash
docker compose up -d
```

Verify:

```bash
curl http://<VPS_IP>:8000/api/v1/heartbeat     # → "pong"
```

---

## 3 ️⃣ 12 : 20 – 12 : 40 🗄️ Chromadb quick-test (GPU-box → VPS)

Install client on GPU-box env:

```bash
pip install chromadb
python - <<'PY'
import chromadb
client = chromadb.HttpClient(host="<VPS_IP>", port=8000)
client.create_collection("docs_v1")
print(client.list_collections())
PY
```

You should see `docs_v1` created remotely.

---

## 4 ️⃣ 12 : 40 – 13 : 30 ⚙️ Baseline MS-MARCO training (GPU)

#### 4.1 Prepare config

`configs/msmarco_gpu.yml`

```yaml
data: data/processed/classic_triplets.parquet   # existing
device: cuda
tokeniser: {type: char, max_len: 64}
embedding: {type: lookup, emb_dim: 64}
encoder: {arch: mean, hidden_dim: 128}
batch: 512
epochs: 2                # short run to finish before 15 : 00
lr: 1e-3
wandb_group: "baseline_gpu"
```

*(optional) sample 20 k triplets to speed-train:*

```bash
python tools/sample_dataset.py \
       --in data/processed/classic_triplets.parquet \
       --out data/processed/classic_20k.parquet --rows 20000
```
then change `data:` path.

#### 4.2 Run

```bash
python -m twotower.train --config configs/msmarco_gpu.yml \
      --wandb_project two-tower --wandb
```

Watch W&B dashboard – ensure loss decreases.

*(GPU training runs in foreground ~35 m; you can open a second shell for next tasks.)*

---

## 5 ️⃣ 13 : 30 – 13 : 45 📦 Push checkpoint to Hugging Face

After first epoch finishes (`checkpoints/best_model.pt` appears):

```bash
# assuming HF CLI already authed (huggingface-cli login)
cd checkpoints
huggingface-cli repo create twotower-char-emb --type=model -y
git init && git remote add origin https://huggingface.co/azuremis/twotower-char-emb
git lfs track "*.pt"
git add best_model.pt .gitattributes
git commit -m "baseline char embedding"
git push origin main
```

Backups now safe & downloadable by inference container.

---

## 6 ️⃣ 13 : 45 – 14 : 10 📥 Index document tower into Chromadb

Create `inference/index_to_chroma.py`

```python
import argparse, torch, pandas as pd, chromadb, yaml
from twotower.encoders import MeanTower
from twotower.embeddings import LookUpEmbedding
from twotower.tokenisers import CharTokeniser

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--triplets", required=True)
parser.add_argument("--host", default="<VPS_IP>")
parser.add_argument("--collection", default="docs_v1")
args = parser.parse_args()

trip = pd.read_parquet(args.triplets)["d_pos_text"].drop_duplicates()
tok = CharTokeniser(); tok.fit(trip.tolist())
vocab_size = len(tok.t2i)+1
emb = LookUpEmbedding(vocab_size, 64)
enc = MeanTower(emb, 128)
enc.load_state_dict(torch.load(args.model)["model"], strict=False)
enc.eval().cuda()

def batch(iterable, n=1024):
    l=len(iterable)
    for i in range(0,l,n): yield iterable[i:i+n]

client = chromadb.HttpClient(host=args.host, port=8000)
col = client.get_or_create_collection(args.collection)

with torch.no_grad():
  for chunk in batch(trip.tolist(), 512):
      ids = [str(hash(t)) for t in chunk]
      toks = torch.tensor([tok.pad(tok.encode(t),64) for t in chunk]).cuda()
      embs = enc(toks).cpu().numpy()
      col.upsert(ids=ids, embeddings=embs, metadatas=[{"text":t} for t in chunk])
```

Run:

```bash
python inference/index_to_chroma.py \
       --model checkpoints/best_model.pt \
       --triplets data/processed/classic_triplets.parquet
```

You will see upsert logs; 20 k docs finish in seconds.

---

## 7 ️⃣ 14 : 10 – 14 : 25 🔎 End-to-end smoke test

```bash
python inference/cli/retrieve.py \
       --query "how many calories in an egg" \
       --backend chroma \
       --host <VPS_IP> --top_k 5
```

Top results should show calorie passages. ✅

---

## 8 ️⃣ 14 : 25 – 14 : 55 📊 Experiment grid / sweep script

`tools/sweep.py`

```python
import subprocess, itertools, pathlib, yaml

configs = ["char_tower.yml", "word2vec_skipgram.yml", "glove_search.yml"]
for cfg in configs:
    cfg_path = f"configs/{cfg}"
    run_name = pathlib.Path(cfg).stem
    subprocess.run([
        "python", "-m", "twotower.train",
        "--config", cfg_path,
        "--wandb", "--wandb_project", "two-tower",
        "--wandb_run_name", run_name,
        "--device", "cuda"
    ])
```

Run as a background `nohup` (after 15 : 00). W&B groups runs for easy comparison.

---

## 9 ️⃣ 14 : 55 – 15 : 00 💾 Commit & TODO log

```bash
git add inference/index_to_chroma.py tools/sweep.py configs/msmarco_gpu.yml
git commit -m "GPU baseline + chroma indexing + sweep script"
git push
```

Create `TODO_EVENING.md`

```markdown
* run sweep.py overnight
* add ANN (chroma + HNSW) for >1M docs
* finish docker/inference Dockerfile:
  FROM python:3.11-slim
  RUN pip install twotower chromadb[client] huggingface_hub fastapi uvicorn
  CMD ["python", "inference/api.py"]
* Optuna hyper-tune learning rate, hidden_dim
* EDA notebook: compare triplet variants
```

---

## Deployment map

| Component | Host | Runtime | Dockerised? |
|-----------|------|---------|-------------|
| **Training jobs** | **GPU-box** | Conda env, bare Python | ❌ (no need) |
| **Chromadb vector store** | **VPS** | Docker (`chromadb` image) | ✅ |
| **Inference API** | **VPS** | Docker (build from repo) | ✅ |
| **Weights & Biases** | Cloud | SaaS | – |
| **Models/Embeddings** | Hugging Face Hub | Git-LFS | – |

---

## Copy-paste snippets for future AI-assist

### Install chromadb client
```bash
pip install "chromadb[client]"
```

### Query collection
```python
client = chromadb.HttpClient(host="VPS_IP", port=8000)
col = client.get_collection("docs_v1")
res = col.query(query_embeddings=[q_vec], n_results=5, include=["metadatas"])
```

### FastAPI micro-service (inference/api.py)
```python
from fastapi import FastAPI
import chromadb, torch, yaml, os
from twotower import load_model_and_tokeniser  # helper you’ll write

app = FastAPI()
client = chromadb.HttpClient(host=os.environ["CHROMA_HOST"], port=8000)
col = client.get_collection("docs_v1")
model, tok = load_model_and_tokeniser(os.environ["MODEL_REPO_URL"])

@app.get("/search")
def search(q: str, k: int = 5):
    ids = tok.pad(tok.encode(q), 64)
    with torch.no_grad():
        q_vec = model.q(torch.tensor([ids]).cuda()).cpu().numpy()
    out = col.query(query_embeddings=q_vec, n_results=k, include=["metadatas"])
    return out["metadatas"][0]
```

---

### ⏱️ Buffer time (5 min)

Use it to troubleshoot any step lagging. If everything green-lights early, start Dockerfile for inference.

---

Made to be **follow-by-copy-paste** — no external googling required. Happy shipping!
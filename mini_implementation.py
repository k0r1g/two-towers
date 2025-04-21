#
#
#
"""
two_tower.py
============

Minimal implementation of a Two‑Tower (Dual Encoder) neural network for
document retrieval.

Usage
-----
$ python two_tower.py --data pairs.tsv --epochs 3

The data file `pairs.tsv` must contain three tab‑separated columns:

    query<TAB>document<TAB>label

where *label* is 1 for a relevant (positive) pair and 0 for a random
(negative) pair.

The script will:
  • build a character‑level vocabulary (easy to inspect)
  • train the model with in‑batch negative sampling
  • write `model.pt` – ready for loading and inference

Tested with Python 3.11 and PyTorch 2.2.
"""

#
#
#
import argparse, collections, random, pickle, math, os, sys, time
import torch, torch.nn.functional as F
from torch import nn
from tqdm import tqdm


#
# Dataset & tokeniser
#
class CharVocab:
  """Character to integer lookup tables."""
  def __init__(self, texts):
    chars = sorted({c for t in texts for c in t})
    self.stoi = {c:i+1 for i,c in enumerate(chars)} # 0 = padding
    self.itos = {i:c for c,i in self.stoi.items()}

  def encode(self, text): return [self.stoi.get(c, 0) for c in text]
  def decode(self, ids):  return ''.join(self.itos.get(i, '?') for i in ids)
  def __len__(self):      return len(self.stoi) + 1 # + pad


class Pairs(torch.utils.data.Dataset):
  """<query, doc, label> triples."""
  def __init__(self, path, vocab=None, max_len=64):
    raw = [l.rstrip('\n').split('\t') for l in open(path)]
    q, d, y = zip(*raw)
    self.vocab = vocab if vocab else CharVocab(q+d)
    self.queries = [self.truncate(self.vocab.encode(t), max_len) for t in q]
    self.docs    = [self.truncate(self.vocab.encode(t), max_len) for t in d]
    self.labels  = [int(i) for i in y]

  def truncate(self, seq, n): return seq[:n] + [0]*(n-len(seq)) if len(seq)<n else seq[:n]
  def __len__(self): return len(self.labels)
  def __getitem__(self, i):
    return (torch.tensor(self.queries[i]), torch.tensor(self.docs[i]), torch.tensor(self.labels[i], dtype=torch.float32))


#
# Two‑Tower encoder
#
class Tower(nn.Module):
  def __init__(self, vocab, emb=64, hid=128):
    super().__init__()
    self.emb = nn.Embedding(vocab, emb, padding_idx=0)
    self.ff  = nn.Sequential(
        nn.Linear(emb, hid),
        nn.ReLU(),
        nn.Linear(hid, hid))
  def forward(self, x):
    # x: (B,L)
    mask = (x>0).float().unsqueeze(-1)
    emb  = self.emb(x) * mask                # (B,L,E)
    emb  = emb.sum(1) / (mask.sum(1)+1e-9)   # mean pooling -> (B,E)
    return F.normalize(self.ff(emb), dim=-1) # unit vectors


class TwoTower(nn.Module):
  def __init__(self, vocab, emb=64, hid=128):
    super().__init__()
    self.q = Tower(vocab, emb, hid)
    self.d = Tower(vocab, emb, hid)

  def forward(self, q, d):
    return self.q(q), self.d(d)              # (B,H), (B,H)


#
# Training utilities
#
def train(model, data, *, epochs=3, lr=1e-3, bs=256, device='cpu'):
  model.to(device)
  opt = torch.optim.AdamW(model.parameters(), lr=lr)
  loader = torch.utils.data.DataLoader(data, batch_size=bs, shuffle=True)

  for ep in range(1, epochs+1):
    tot, n = 0.0, 0
    for q,d,y in tqdm(loader, desc=f'Epoch {ep}', ncols=80):
      q,d = q.to(device),d.to(device)
      qv,dv = model(q,d)             # (B,H)
      sims = qv @ dv.t()             # (B,B) dot product
      lbls = torch.arange(len(qv), device=device)
      loss = F.cross_entropy(sims, lbls)     # in-batch negatives
      opt.zero_grad()
      loss.backward()
      opt.step()
      tot += loss.item()*len(qv); n += len(qv)
    print(f'  mean loss {tot/n:.4f}')
  return model


#
# Simple retrieval demo
#
@torch.no_grad()
def retrieve(model, query, docs, vocab, topk=5, device='cpu'):
  q = torch.tensor([vocab.encode(query)], device=device)
  d = torch.tensor([vocab.encode(t) for t in docs], device=device)
  qv, dv = model.q(q), model.d(d)
  scores = (qv @ dv.t()).squeeze(0)   # (N,)
  idx = scores.argsort(descending=True)[:topk]
  return [(docs[i], scores[i].item()) for i in idx]


#
# CLI
#
def main():
  p = argparse.ArgumentParser()
  p.add_argument('--data', required=True, help='pairs.tsv')
  p.add_argument('--epochs', type=int, default=3)
  p.add_argument('--bs', type=int, default=256)
  p.add_argument('--device', default='cpu')
  args = p.parse_args()

  ds = Pairs(args.data)
  model = TwoTower(len(ds.vocab))
  model = train(model, ds, epochs=args.epochs, bs=args.bs, device=args.device)
  torch.save({'model':model.state_dict(),
              'vocab':ds.vocab.stoi}, 'model.pt')
  print('model saved to model.pt')

  # quick sanity check
  print('\nSample retrieval:')
  doc_texts = [d for _,d,_ in random.sample(list(zip(ds.queries, ds.docs, ds.labels)), 20)]
  query = input('Type a query: ')
  results = retrieve(model, query, doc_texts, ds.vocab)
  for txt,score in results: print(f'  {score:+.3f} | {txt[:80]}...')


if __name__ == '__main__':
  main()

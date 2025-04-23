import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# 🧠 Query Tower
# Two-layer MLP with ReLU to transform query embeddings
# ─────────────────────────────────────────────────────────────────────────────
class QryTower(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )

    def forward(self, x):
        return self.fc(x)

# ─────────────────────────────────────────────────────────────────────────────
# 📄 Document Tower
# Same structure as QryTower — can be customized independently
# ─────────────────────────────────────────────────────────────────────────────
class DocTower(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )

    def forward(self, x):
        return self.fc(x)

# ─────────────────────────────────────────────────────────────────────────────
# 📚 CBOW (Continuous Bag of Words) Word2Vec Model
# Trained using context words → target word prediction
# ─────────────────────────────────────────────────────────────────────────────
class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.ffw = nn.Linear(in_features=embed_dim, out_features=vocab_size, bias=False)

    def forward(self, x):
        emb = self.emb(x)              # (batch_size, context_window, embed_dim)
        pooled = emb.mean(dim=1)       # (batch_size, embed_dim)
        out = self.ffw(pooled)         # (batch_size, vocab_size)
        return out

# # ─────────────────────────────────────────────────────────────────────────────
# # 🔬 Quick test for CBOW forward + loss
# # ─────────────────────────────────────────────────────────────────────────────
# if __name__ == '__main__':
#     model = CBOW(vocab_size=128, embed_dim=8)
#     print('🧠 CBOW:', model)

#     criterion = nn.CrossEntropyLoss()
#     inpt = torch.randint(0, 128, (3, 5))  # (batch_size, context_window)
#     trgt = torch.randint(0, 128, (3,))    # (batch_size)

#     out = model(inpt)
#     loss = criterion(out, trgt)
#     print("📉 Loss:", loss.item())  # ~ ln(1/128) → ~4.85 for random init
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import wandb

# Init W&B
wandb.init(
    project="two-tower-ms_marco",
    entity="dtian09",
    config={
        "hidden_dim": 128,
        "batch_size": 32,
        "epochs": 20,
        "max_query_len": 20,
        "max_passage_len": 200,
        "margin": 0.2,
        "lr": 1e-3,
        "patience": 3
    }
)
config = wandb.config

# Text to embedding
def text_to_embedding(text, word_to_id, embedding_matrix, max_len, unk_id=0):
    tokens = text.lower().split()
    ids = [word_to_id.get(tok, unk_id) for tok in tokens]
    ids = ids[:max_len] + [unk_id] * max(0, max_len - len(ids))
    embeddings = torch.stack([embedding_matrix[i] for i in ids])
    return embeddings

# Dataset class
class TripletDataset(Dataset):
    def __init__(self, dataset, word_to_id, embedding_matrix, max_query_len=20, max_passage_len=200):
        self.data = []
        self.embedding_matrix = embedding_matrix
        self.word_to_id = word_to_id
        self.max_query_len = max_query_len
        self.max_passage_len = max_passage_len

        for item in dataset:
            query = item["query"]
            passages = item["passages"]
            pos = [p["passage_text"] for p in passages if p["is_selected"] == 1]
            neg = [p["passage_text"] for p in passages if p["is_selected"] == 0]
            for p, n in zip(pos, neg):
                self.data.append((query, p, n))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query, pos, neg = self.data[idx]
        q_embed = text_to_embedding(query, self.word_to_id, self.embedding_matrix, self.max_query_len)
        p_pos = text_to_embedding(pos, self.word_to_id, self.embedding_matrix, self.max_passage_len)
        p_neg = text_to_embedding(neg, self.word_to_id, self.embedding_matrix, self.max_passage_len)
        return q_embed, p_pos, p_neg

def collate_batch(batch):
    q, p, n = zip(*batch)
    return torch.stack(q), torch.stack(p), torch.stack(n)

# Model
class SimpleRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.predictor = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, token_embedding, hidden_state):
        combined = torch.cat((token_embedding, hidden_state), dim=-1)
        hidden_state = torch.tanh(self.linear(combined))
        prediction = self.predictor(hidden_state)
        return hidden_state, prediction

class TwoTowerTripletRNN(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.query_encoder = SimpleRNNCell(embed_dim, hidden_dim)
        self.passage_encoder = SimpleRNNCell(embed_dim, hidden_dim)

    def encode_sequence(self, embeddings, rnn_cell):
        batch_size, seq_len, _ = embeddings.shape
        hidden_state = torch.zeros(batch_size, rnn_cell.predictor.out_features, device=embeddings.device)
        for t in range(seq_len):
            token_embedding = embeddings[:, t, :]
            hidden_state, _ = rnn_cell(token_embedding, hidden_state)
        return hidden_state

    def forward(self, query_embs, pos_embs, neg_embs):
        q_vec = self.encode_sequence(query_embs, self.query_encoder)
        p_pos = self.encode_sequence(pos_embs, self.passage_encoder)
        p_neg = self.encode_sequence(neg_embs, self.passage_encoder)
        return q_vec, p_pos, p_neg

def triplet_loss(q_vec, pos_vec, neg_vec, margin=0.2):
    sim_pos = F.cosine_similarity(q_vec, pos_vec, dim=-1)
    sim_neg = F.cosine_similarity(q_vec, neg_vec, dim=-1)
    return F.relu(margin - sim_pos + sim_neg).mean()

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for query, pos, neg in loader:
            query, pos, neg = query.to(device), pos.to(device), neg.to(device)
            q_vec, p_pos, p_neg = model(query, pos, neg)
            sim_pos = F.cosine_similarity(q_vec, p_pos, dim=-1)
            sim_neg = F.cosine_similarity(q_vec, p_neg, dim=-1)
            correct += (sim_pos > sim_neg).sum().item()
            total += query.size(0)
    return correct / total if total > 0 else 0

def train_validate_test():
    # Load dataset and embeddings
    from huggingface_hub import hf_hub_download
    import torch

    repo_id = "dtian09/MS_MARCO"
    skip_gram_path = hf_hub_download(repo_id=repo_id, filename="skip_gram_model.pt")
    word_to_id_path = hf_hub_download(repo_id=repo_id, filename="word_to_id.pt")

    state_dict = torch.load(skip_gram_path, map_location="cpu")
    word_to_id = torch.load(word_to_id_path)
    embedding_matrix = state_dict["target_embedding.weight"]
    embed_dim = embedding_matrix.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset splits
    train_data = load_dataset("microsoft/ms_marco", "v1.1", split="train")
    val_data = load_dataset("microsoft/ms_marco", "v1.1", split="validation")

    train_dataset = TripletDataset(train_data, word_to_id, embedding_matrix, config.max_query_len, config.max_passage_len)
    val_dataset = TripletDataset(val_data, word_to_id, embedding_matrix, config.max_query_len, config.max_passage_len)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch)
    
    model = TwoTowerTripletRNN(embed_dim=embed_dim, hidden_dim=config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_val_acc = 0
    patience = config.patience
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for query, pos, neg in train_loader:
            query, pos, neg = query.to(device), pos.to(device), neg.to(device)
            q_vec, p_pos, p_neg = model(query, pos, neg)
            loss = triplet_loss(q_vec, p_pos, p_neg, margin=config.margin)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader, device)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_accuracy": val_acc
        })

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Accuracy: {val_acc:.4f}")

        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_two_tower_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    # Load best model for final evaluation
    model.load_state_dict(torch.load("best_two_tower_model.pt"))
    test_data = load_dataset("microsoft/ms_marco", "v1.1", split="test")
    test_dataset = TripletDataset(test_data, word_to_id, embedding_matrix, config.max_query_len, config.max_passage_len)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch)
    test_acc = evaluate(model, test_loader, device)
    wandb.log({"test_accuracy": test_acc})
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    train_validate_test()
    

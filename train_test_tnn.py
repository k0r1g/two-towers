'''
dual encoder model (two tower neural network) with triplet loss
input: query, positive passage, negative passage
output: similarity score between query and positive passage, similarity score between query and negative passage
training loss: triplet loss
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from huggingface_hub import hf_hub_download

# ------------------ Load Files ------------------
def load_pt_files():
    repo_id = "dtian09/MS_MARCO"

    skip_gram_path = hf_hub_download(repo_id=repo_id, filename="skip_gram_model.pt")
    word_to_id_path = hf_hub_download(repo_id=repo_id, filename="word_to_id.pt")
    query2passage_path = hf_hub_download(repo_id=repo_id, filename="query2passage.pt")

    state_dict = torch.load(skip_gram_path)
    word_to_id = torch.load(word_to_id_path)
    query2passage = torch.load(query2passage_path)

    embedding_matrix = state_dict["target_embedding.weight"]
    return embedding_matrix, word_to_id, query2passage

# ------------------ Text to Embedding ------------------
def text_to_embedding(text, word_to_id, embedding_matrix, max_len, unk_id=0):
    tokens = text.lower().split()
    ids = [word_to_id.get(tok, unk_id) for tok in tokens]
    ids = ids[:max_len] + [unk_id] * max(0, max_len - len(ids))
    embeddings = torch.stack([embedding_matrix[i] for i in ids])
    return embeddings  # (seq_len, embed_dim)

# ------------------ Triplet Dataset ------------------
class TripletDataset(Dataset):
    def __init__(self, query2passage, word_to_id, embedding_matrix,
                 max_query_len=20, max_passage_len=200):
        self.data = []
        self.embedding_matrix = embedding_matrix
        self.word_to_id = word_to_id
        self.max_query_len = max_query_len
        self.max_passage_len = max_passage_len

        for query, (pos_list, neg_list) in query2passage.items():
            for pos, neg in zip(pos_list, neg_list):
                self.data.append((query, pos, neg))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query, pos, neg = self.data[idx]
        q_embed = text_to_embedding(query, self.word_to_id, self.embedding_matrix, self.max_query_len)
        p_pos = text_to_embedding(pos, self.word_to_id, self.embedding_matrix, self.max_passage_len)
        p_neg = text_to_embedding(neg, self.word_to_id, self.embedding_matrix, self.max_passage_len)
        return q_embed, p_pos, p_neg

# Collate batch
def collate_batch(batch):
    q, p, n = zip(*batch)
    return torch.stack(q), torch.stack(p), torch.stack(n)

# ------------------ Model & Loss ------------------
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

    accuracy = correct / total if total > 0 else 0
    return accuracy

# ------------------ Training Loop ------------------
def train():
    embedding_matrix, word_to_id, query2passage = load_pt_files()
    embed_dim = embedding_matrix.shape[1]
    hidden_dim = 128
    batch_size = 32
    epochs = 5
    max_query_len = 20
    max_passage_len = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = TripletDataset(
        query2passage=query2passage,
        word_to_id=word_to_id,
        embedding_matrix=embedding_matrix,
        max_query_len=max_query_len,
        max_passage_len=max_passage_len
    )

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    model = TwoTowerTripletRNN(embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for query, pos, neg in train_loader:
            query, pos, neg = query.to(device), pos.to(device), neg.to(device)
            q_vec, p_pos, p_neg = model(query, pos, neg)
            loss = triplet_loss(q_vec, p_pos, p_neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} - Test Accuracy: {test_acc:.4f}")

# NOTE: To run training with evaluation, call train() in your local environment.
# train()

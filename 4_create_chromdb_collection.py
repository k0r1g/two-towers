'''
This script creates a ChromaDB collection from the MS MARCO passage dataset.
It loads the dataset, encodes the passages using a SentenceTransformer model,
and stores the embeddings in ChromaDB.
'''
import chromadb
from chromadb.config import Settings
from datasets import load_dataset
#from sentence_transformers import SentenceTransformer
import uuid
import torch

# === Settings ===
COLLECTION_NAME = "ms_marco_passages"
N_PASSAGES = 1000  # Load more if you want
MAX_BATCH_SIZE = 5000  # Stay safely under ChromaDB's max
PERSIST_DIR = "./chroma_db"  # <-- Add this line for persistence

# === Load MS MARCO Dataset ===
print("Loading MS MARCO dataset...")
dataset = load_dataset("ms_marco", "v1.1", split="train").select(range(N_PASSAGES))
passagesL = []
for item in dataset:
    passages = item["passages"]
    ps = passages['passage_text']
    passagesL.extend(ps)

#tokenize the passages and look up embeddings for each passage from the embedding matrix
import preprocess as pre
from huggingface_hub import hf_hub_download

repo_id = "dtian09/MS_MARCO"
skip_gram_path = hf_hub_download(repo_id=repo_id, filename="skip_gram_model.pt",  repo_type="dataset" )
word_to_id_path = hf_hub_download(repo_id=repo_id, filename="word_to_id.pt",  repo_type="dataset" )
state_dict = torch.load(skip_gram_path, map_location="cpu")
word_to_id = torch.load(word_to_id_path)
embedding_matrix = state_dict["target_embedding.weight"]
embed_dim = embedding_matrix.shape[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load('best_two_tower_model_test_accuracy_0_51.pt', map_location="cpu")

# Get hidden_dim from the RNN weight
rnn_weight = state_dict['passage_encoder.predictor.weight']

if rnn_weight.shape[0] % 4 == 0:
    hidden_dim = rnn_weight.shape[0] // 4  # likely an LSTM
else:
    hidden_dim = rnn_weight.shape[0]       # likely a GRU or simple RNN

from train_test_tnn import TwoTowerTripletRNN

model = TwoTowerTripletRNN(embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
model.load_state_dict(state_dict)
model.eval()

encoded_passagesL = []
for passage in passagesL:
    embeddings=pre.text_to_embedding(passage, word_to_id, embedding_matrix, unk_id=0)
    encoded_passage = model.encode_sequence(embeddings, model.passage_encoder)
    encoded_passagesL.append(encoded_passage)

# === Load SentenceTransformer Encoder ===
#print("Encoding passages...")
#encoder = SentenceTransformer("all-MiniLM-L6-v2")
#embeddings = encoder.encode(texts, convert_to_numpy=True, show_progress_bar=True)

from chromadb import PersistentClient

PERSIST_DIR = "./chroma_db"

# === Initialize ChromaDB with Persistence ===
client = PersistentClient(path=PERSIST_DIR)

# Create or clear collection
if COLLECTION_NAME in [c.name for c in client.list_collections()]:
    client.delete_collection(name=COLLECTION_NAME)
collection = client.create_collection(name=COLLECTION_NAME)

# === Add to ChromaDB in batches ===
print("Adding to ChromaDB...")

def chunked(data, size):
    for i in range(0, len(data), size):
        yield data[i:i + size]

for doc_batch, emb_batch in zip(
    chunked(passagesL, MAX_BATCH_SIZE),
    chunked(encoded_passagesL, MAX_BATCH_SIZE)
):
    collection.add(
        documents=doc_batch,
        embeddings=emb_batch,
        ids=[str(uuid.uuid4()) for _ in doc_batch],
        metadatas=[{"source": "ms_marco"} for _ in doc_batch]
    )

print(f"Collection '{COLLECTION_NAME}' created with {len(texts)} passages.")
print("ChromaDB is ready for querying!")
# Note: You can now run the search engine script to query this collection.

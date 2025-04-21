# 00_marco_tokeniser.py – Tokenise MS MARCO queries + passages
# -------------------------------------------------------------
import collections
import pickle
import random
import re
import os
import sys
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

# Add wandb for tracking
import wandb
from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# ---------------------------------------------------------------------------
# 0️⃣  Parameters + NLTK setup
# ---------------------------------------------------------------------------
VERSION        = "v1.1"     # or v2.1
NUM_EXAMPLES   = 5000       # set to None for full dataset
TOP_N_WORDS    = 50000
PAD_TOKEN      = "<PAD>"
SAVE_DIR       = Path(".")

# Initialize wandb
print("🔄 Initializing wandb tracking...")
# Get wandb API key from environment (using the correct variable name)
wandb_key = os.environ.get("WANDDB_KEY")  # Note: Using WANDDB_KEY as specified
if not wandb_key:
    print("Warning: WANDDB_KEY not found in .env file")
    wandb_key = input("Enter your Weights & Biases API key: ")
    os.environ["WANDB_API_KEY"] = wandb_key
else:
    # Set the environment variable that wandb actually uses
    os.environ["WANDB_API_KEY"] = wandb_key

# Start a new wandb run
run = wandb.init(
    # Set entity (your team name) if needed
    # entity="your-team-name",  # Uncomment and set if needed
    project="ms-marco-tokenizer",
    # Track parameters
    config={
        "version": VERSION,
        "num_examples": NUM_EXAMPLES,
        "top_n_words": TOP_N_WORDS,
    },
)
print("✅ Tracking initialized")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
TOKEN_RE   = re.compile(r"[^\w\s-]")

# ---------------------------------------------------------------------------
# 1️⃣  Load MS MARCO dataset
# ---------------------------------------------------------------------------
print("📦 Loading MS MARCO dataset …")
dataset = load_dataset("microsoft/ms_marco", VERSION, split="train")
print(f"✔ Loaded split: {len(dataset):,} examples")

# Select a random subset
if NUM_EXAMPLES:
    dataset = dataset.shuffle(seed=42).select(range(NUM_EXAMPLES))
    print(f"🔎 Subsampled to {NUM_EXAMPLES:,} rows")

# ---------------------------------------------------------------------------
# 2️⃣  Preprocessing
# ---------------------------------------------------------------------------
def preprocess(text: str) -> list[str]:
    """Lowercase, remove punctuation, tokenize, drop stopwords, keep alpha."""
    text = text.lower()
    text = TOKEN_RE.sub(" ", text).replace("-", " ")
    return [w for w in word_tokenize(text) if w.isalpha() and w not in STOP_WORDS]

# ---------------------------------------------------------------------------
# 3️⃣  Tokenise queries + relevant/irrelevant passages
# ---------------------------------------------------------------------------
query_tokens         = []
relevant_pass_tokens = []
irrelevant_pass_tokens = []
all_tokens           = []

# Create a progress counter for wandb tracking
processed_count = 0
total_count = len(dataset)
run.log({"total_examples": total_count})

for item in dataset:
    query = item["query"]
    passages = item["passages"]
    
    # Tokenise query
    qtoks = preprocess(query)
    query_tokens.append(qtoks)
    all_tokens.extend(qtoks)

    # Iterate over the passage texts and is_selected values using zip
    for i, (passage_text, is_selected) in enumerate(zip(passages["passage_text"], passages["is_selected"])):
        ptoks = preprocess(passage_text)
        all_tokens.extend(ptoks)
        
        if is_selected == 1:
            relevant_pass_tokens.append(ptoks)
        else:
            irrelevant_pass_tokens.append(ptoks)
            
    # Update progress every 100 examples
    processed_count += 1
    if processed_count % 100 == 0:
        run.log({
            "processed_examples": processed_count,
            "progress_percent": (processed_count / total_count) * 100
        })

print(f"🧼 Total tokens: {len(all_tokens):,}")
print("🧪 Sample query tokens:", query_tokens[0])
print("🧪 Sample relevant passage tokens:", relevant_pass_tokens[0])

run.log({
    "total_tokens": len(all_tokens),
    "unique_tokens": len(set(all_tokens)),
    "num_queries": len(query_tokens),
    "num_relevant_passages": len(relevant_pass_tokens),
    "num_irrelevant_passages": len(irrelevant_pass_tokens),
})

# ---------------------------------------------------------------------------
# 4️⃣  Build vocabulary (TOP_N_WORDS most common)
# ---------------------------------------------------------------------------
freq = collections.Counter(all_tokens)
most_common = [w for w, _ in freq.most_common(TOP_N_WORDS)]

idx_to_word = [PAD_TOKEN] + most_common
word_to_idx = {w: i for i, w in enumerate(idx_to_word)}

print(f"🧠 Vocabulary size (incl. PAD): {len(idx_to_word):,}")
run.log({"vocabulary_size": len(idx_to_word)})

# ---------------------------------------------------------------------------
# 5️⃣  Convert tokens to IDs
# ---------------------------------------------------------------------------
def tokens_to_ids(token_lists, word_to_idx):
    return [[word_to_idx.get(w, 0) for w in toks] for toks in token_lists]

query_token_ids         = tokens_to_ids(query_tokens, word_to_idx)
relevant_pass_token_ids = tokens_to_ids(relevant_pass_tokens, word_to_idx)
irrelevant_pass_token_ids = tokens_to_ids(irrelevant_pass_tokens, word_to_idx)
corpus_ids              = [word_to_idx.get(w, 0) for w in all_tokens]

# ---------------------------------------------------------------------------
# 6️⃣  Save artifacts
# ---------------------------------------------------------------------------
print("💾 Saving tokenisation artifacts …")
pickle.dump(all_tokens,                open(SAVE_DIR / "corpus.pkl", "wb"))
pickle.dump(query_tokens,             open(SAVE_DIR / "query_tokens.pkl", "wb"))
pickle.dump(relevant_pass_tokens,     open(SAVE_DIR / "relevant_passages.pkl", "wb"))
pickle.dump(irrelevant_pass_tokens,   open(SAVE_DIR / "irrelevant_passages.pkl", "wb"))

pickle.dump(query_token_ids,          open(SAVE_DIR / "query_token_ids.pkl", "wb"))
pickle.dump(relevant_pass_token_ids,  open(SAVE_DIR / "relevant_token_ids.pkl", "wb"))
pickle.dump(irrelevant_pass_token_ids,open(SAVE_DIR / "irrelevant_token_ids.pkl", "wb"))

pickle.dump(word_to_idx,              open(SAVE_DIR / "word_to_idx.pkl", "wb"))
pickle.dump(idx_to_word,              open(SAVE_DIR / "idx_to_word.pkl", "wb"))

# Log final metrics to wandb
run.log({
    "completed": True,
    "files_saved": 9
})

print("✅ Done. Artifacts:")
for f in [
    "corpus.pkl", "query_tokens.pkl", "relevant_passages.pkl", "irrelevant_passages.pkl",
    "query_token_ids.pkl", "relevant_token_ids.pkl", "irrelevant_token_ids.pkl",
    "word_to_idx.pkl", "idx_to_word.pkl"
]:
    print("•", f)

# ---------------------------------------------------------------------------
# 7️⃣  Upload to Hugging Face Hub
# ---------------------------------------------------------------------------
print("\nUploading to Hugging Face Hub …")
try:
    from huggingface_hub import HfApi
except ImportError:
    os.system(f"{sys.executable} -m pip install -q huggingface_hub")
    from huggingface_hub import HfApi

api = HfApi()
# Get Hugging Face token from environment (using the correct variable name)
hf_token = os.environ.get("HUGGINGFACE_KEY")  # Note: Using HUGGINGFACE_KEY as specified
if not hf_token:
    print("Warning: HUGGINGFACE_KEY not found in .env file")
    hf_token = input("Enter your Hugging Face access token: ")
    
repo_id = "Kogero/ms-marco-tokenized"  # Using your actual HF username
api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, token=hf_token)

files_to_upload = {
    "corpus.pkl": "data/corpus.pkl",
    "query_tokens.pkl": "data/query_tokens.pkl",
    "relevant_passages.pkl": "data/relevant_passages.pkl",
    "irrelevant_passages.pkl": "data/irrelevant_passages.pkl",
    "query_token_ids.pkl": "data/query_token_ids.pkl",
    "relevant_token_ids.pkl": "data/relevant_token_ids.pkl",
    "irrelevant_token_ids.pkl": "data/irrelevant_token_ids.pkl",
    "word_to_idx.pkl": "tokenizer/word_to_idx.pkl",
    "idx_to_word.pkl": "tokenizer/idx_to_word.pkl",
}

for local, remote in files_to_upload.items():
    print(f"  ↳ uploading {local} → {remote}")
    api.upload_file(
        path_or_fileobj=str(SAVE_DIR / local),
        path_in_repo=remote,
        repo_id=repo_id,
        repo_type="dataset",
        token=hf_token
    )

print(f"\n✅  Done – dataset at https://huggingface.co/datasets/{repo_id}")

# Finish wandb run at the very end
run.finish()

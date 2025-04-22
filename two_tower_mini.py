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
  • train the model with triplet margin loss
  • write `model.pt` – ready for loading and inference

Tested with Python 3.11 and PyTorch 2.2.
"""

#
#
#
import argparse, collections, random, pickle, math, os, sys, time, datetime
import torch, torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import pandas as pd
import wandb
from dotenv import load_dotenv
import logging
import json
from pprint import pformat
from typing import Dict, List, Tuple, Any, Optional

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('two_tower.log', mode='w')
    ]
)
logger = logging.getLogger('two_tower')

# Helper function to log tensor info
def log_tensor_info(tensor, name="tensor"):
    """Log helpful information about tensors"""
    if isinstance(tensor, torch.Tensor):
        logger.info(f"{name} shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")
        logger.info(f"{name} stats: min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, "
                    f"mean={tensor.float().mean().item():.4f}, std={tensor.float().std().item():.4f}")
        if tensor.numel() < 10:
            logger.info(f"{name} full content: {tensor}")
        else:
            logger.info(f"{name} sample: {tensor.flatten()[:5].tolist()} ... {tensor.flatten()[-5:].tolist()}")
    elif isinstance(tensor, list):
        logger.info(f"{name} type: list, length: {len(tensor)}")
        if len(tensor) < 10:
            logger.info(f"{name} full content: {tensor}")
        else:
            logger.info(f"{name} sample: {tensor[:3]} ... {tensor[-3:]}")
    else:
        logger.info(f"{name}: {tensor}")

# Ensure checkpoints directory exists
os.makedirs('checkpoints', exist_ok=True)

#
# Dataset & tokeniser
#
class CharVocab:
  """Character to integer lookup tables."""
  def __init__(self, texts):
    logger.info(f"Building vocabulary from {len(texts)} texts")
    logger.info(f"Sample texts: {texts[:3]}")
    
    # Get all unique characters
    chars = sorted({char for text in texts for char in text})
    logger.info(f"Found {len(chars)} unique characters: {chars[:50]}{'...' if len(chars) > 50 else ''}")
    
    self.string_to_index = {char: idx+1 for idx, char in enumerate(chars)} # 0 = padding
    self.index_to_string = {idx: char for char, idx in self.string_to_index.items()}
    
    logger.info(f"Vocabulary size (including padding): {len(self)}")
    logger.info(f"Sample mappings: {list(self.string_to_index.items())[:5]}")

  def encode(self, text): 
    encoded = [self.string_to_index.get(char, 0) for char in text]
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Encoded text '{text[:20]}...' to {encoded[:20]}...")
    return encoded
    
  def decode(self, indices):  
    decoded = ''.join(self.index_to_string.get(idx, '?') for idx in indices)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Decoded {indices[:20]}... to '{decoded[:20]}...'")
    return decoded
    
  def __len__(self):      
    return len(self.string_to_index) + 1 # + pad


class Triplets(torch.utils.data.Dataset):
  """<query, positive_doc, negative_doc> triplets."""
  def __init__(self, data_path, vocab=None, max_length=64):
    logger.info(f"Loading dataset from {data_path}")
    
    if data_path.endswith('.parquet'):
      # Read from parquet
      logger.info("Reading parquet format data")
      dataframe = pd.read_parquet(data_path)
      logger.info(f"Dataframe shape: {dataframe.shape}")
      logger.info(f"Dataframe columns: {dataframe.columns.tolist()}")
      logger.info(f"Dataframe sample:\n{dataframe.head(3)}")
      
      queries = dataframe['query'].tolist()
      documents = dataframe['document'].tolist()
      labels = dataframe['label'].tolist()
    else:
      # Legacy TSV format
      logger.info("Reading TSV format data")
      raw_lines = [line.rstrip('\n').split('\t') for line in open(data_path)]
      logger.info(f"Read {len(raw_lines)} lines from TSV file")
      logger.info(f"Sample raw lines: {raw_lines[:3]}")
      
      queries, documents, labels = zip(*raw_lines)
      labels = [int(label) for label in labels]
    
    logger.info(f"Loaded {len(queries)} query-document pairs")
    logger.info(f"Sample queries: {queries[:3]}")
    logger.info(f"Sample documents: {[doc[:50] + '...' for doc in documents[:3]]}")
    logger.info(f"Label distribution: {collections.Counter(labels)}")
    
    # Create vocabulary
    all_texts = queries + documents
    self.vocab = vocab if vocab else CharVocab(all_texts)
    
    # Group queries with positive and negative documents
    logger.info("Grouping queries with positive and negative documents")
    query_to_documents = collections.defaultdict(lambda: {'positive': [], 'negative': []})
    for query, document, label in zip(queries, documents, labels):
      if label == 1:
        query_to_documents[query]['positive'].append(document)
      else:
        query_to_documents[query]['negative'].append(document)
    
    logger.info(f"Created query-document mapping for {len(query_to_documents)} unique queries")
    
    # Log sample of query-document mapping
    sample_queries = list(query_to_documents.keys())[:3]
    for sample_query in sample_queries:
        pos_count = len(query_to_documents[sample_query]['positive'])
        neg_count = len(query_to_documents[sample_query]['negative'])
        logger.info(f"Query: '{sample_query}' has {pos_count} positive and {neg_count} negative documents")
        if pos_count > 0:
            logger.info(f"  Sample positive: '{query_to_documents[sample_query]['positive'][0][:50]}...'")
        if neg_count > 0:
            logger.info(f"  Sample negative: '{query_to_documents[sample_query]['negative'][0][:50]}...'")
    
    # Create triplets
    logger.info("Creating query-positive-negative triplets")
    self.triplets = []
    queries_with_both = 0
    
    for query, docs_dict in query_to_documents.items():
      if docs_dict['positive'] and docs_dict['negative']:  # Only keep queries with both positive and negative docs
        queries_with_both += 1
        for positive_doc in docs_dict['positive']:
          for negative_doc in docs_dict['negative']:
            self.triplets.append((query, positive_doc, negative_doc))
    
    logger.info(f"Created {len(self.triplets)} triplets from {queries_with_both}/{len(query_to_documents)} unique queries with both pos/neg docs")
    
    # Store original texts
    logger.info("Storing original texts and encoding")
    self.query_texts = [triplet[0] for triplet in self.triplets]
    self.positive_doc_texts = [triplet[1] for triplet in self.triplets]
    self.negative_doc_texts = [triplet[2] for triplet in self.triplets]
    
    # Encode texts
    logger.info(f"Max sequence length: {max_length}")
    self.encoded_queries = [self.truncate_and_pad(self.vocab.encode(triplet[0]), max_length) for triplet in self.triplets]
    self.encoded_positive_docs = [self.truncate_and_pad(self.vocab.encode(triplet[1]), max_length) for triplet in self.triplets]
    self.encoded_negative_docs = [self.truncate_and_pad(self.vocab.encode(triplet[2]), max_length) for triplet in self.triplets]
    
    # Log sample triplet
    if self.triplets:
        sample_idx = 0
        logger.info(f"Sample triplet {sample_idx}:")
        logger.info(f"  Query: '{self.query_texts[sample_idx]}'")
        logger.info(f"  Encoded query: {self.encoded_queries[sample_idx][:10]}...")
        logger.info(f"  Positive doc: '{self.positive_doc_texts[sample_idx][:50]}...'")
        logger.info(f"  Encoded positive: {self.encoded_positive_docs[sample_idx][:10]}...")
        logger.info(f"  Negative doc: '{self.negative_doc_texts[sample_idx][:50]}...'")
        logger.info(f"  Encoded negative: {self.encoded_negative_docs[sample_idx][:10]}...")

  def truncate_and_pad(self, sequence, max_length): 
    original_len = len(sequence)
    if len(sequence) < max_length:
      padded = sequence + [0] * (max_length - len(sequence))
      if logger.isEnabledFor(logging.DEBUG):
          logger.debug(f"Padded sequence from length {original_len} to {max_length}")
      return padded
    else:
      truncated = sequence[:max_length]
      if logger.isEnabledFor(logging.DEBUG):
          logger.debug(f"Truncated sequence from length {original_len} to {max_length}")
      return truncated
      
  def __len__(self): 
    return len(self.triplets)
    
  def __getitem__(self, index):
    return (
      torch.tensor(self.encoded_queries[index]), 
      torch.tensor(self.encoded_positive_docs[index]), 
      torch.tensor(self.encoded_negative_docs[index])
    )


#
# Two‑Tower encoder
#
class Tower(nn.Module):
  def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128):
    super().__init__()
    logger.info(f"Initializing Tower with vocab_size={vocab_size}, embedding_dim={embedding_dim}, hidden_dim={hidden_dim}")
    
    self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    self.feed_forward = nn.Sequential(
        nn.Linear(embedding_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim))
    
    # Count parameters
    embedding_params = vocab_size * embedding_dim
    ff_params = embedding_dim * hidden_dim + hidden_dim + hidden_dim * hidden_dim + hidden_dim
    logger.info(f"Tower parameters: embedding={embedding_params:,}, feed_forward={ff_params:,}, total={embedding_params + ff_params:,}")
        
  def forward(self, input_ids):
    # input_ids: (batch_size, sequence_length)
    if logger.isEnabledFor(logging.DEBUG):
        log_tensor_info(input_ids, "Tower input_ids")
    
    mask = (input_ids > 0).float().unsqueeze(-1)
    if logger.isEnabledFor(logging.DEBUG):
        log_tensor_info(mask, "Token mask")
    
    embeddings = self.embedding(input_ids) * mask                # (batch_size, sequence_length, embedding_dim)
    if logger.isEnabledFor(logging.DEBUG):
        log_tensor_info(embeddings, "Embeddings")
    
    pooled = embeddings.sum(1) / (mask.sum(1) + 1e-9)           # mean pooling -> (batch_size, embedding_dim)
    if logger.isEnabledFor(logging.DEBUG):
        log_tensor_info(pooled, "Pooled embeddings")
    
    output = F.normalize(self.feed_forward(pooled), dim=-1)      # unit vectors
    if logger.isEnabledFor(logging.DEBUG):
        log_tensor_info(output, "Tower output")
    
    return output


class TwoTower(nn.Module):
  def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128):
    super().__init__()
    logger.info(f"Initializing TwoTower model with vocab_size={vocab_size}")
    
    self.query_tower = Tower(vocab_size, embedding_dim, hidden_dim)
    self.document_tower = Tower(vocab_size, embedding_dim, hidden_dim)
    
    # Count total parameters
    total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_params:,}")

  def forward(self, query_input, document_input):
    logger.debug("TwoTower forward pass")
    query_vector = self.query_tower(query_input)
    document_vector = self.document_tower(document_input)
    return query_vector, document_vector  # (batch_size, hidden_dim), (batch_size, hidden_dim)


#
# Training utilities
#
def contrastive_triplet_loss(q_emb: torch.Tensor,
                             d_pos_emb: torch.Tensor,
                             d_neg_emb: torch.Tensor,
                             margin: float = 0.2) -> torch.Tensor:
    """
    q_emb, d_pos_emb, d_neg_emb: (B, H) tensors
    margin: the m in max(0, m - cos(q,d+) + cos(q,d-))
    """
    # cosine similarities: (B,)
    sim_pos = F.cosine_similarity(q_emb, d_pos_emb, dim=1)
    sim_neg = F.cosine_similarity(q_emb, d_neg_emb, dim=1)

    # hinge‑margin loss per example
    per_sample_loss = F.relu(margin - sim_pos + sim_neg)

    # average over the batch
    return per_sample_loss.mean()

def train(model, dataset, *, epochs=3, learning_rate=1e-3, batch_size=256, device='cpu', use_wandb=False):
  logger.info(f"Training model for {epochs} epochs with lr={learning_rate}, batch_size={batch_size}, device={device}")
  
  model.to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
  data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
  margin = 0.2
  
  logger.info(f"Dataset size: {len(dataset)} triplets")
  logger.info(f"Batch size: {batch_size}, Number of batches: {len(data_loader)}")
  logger.info(f"Optimizer: {optimizer}")
  logger.info(f"Loss function: contrastive_triplet_loss with margin={margin}")

  # Log model architecture to wandb if enabled
  if use_wandb:
    wandb.watch(model, log_freq=100)
    logger.info("Wandb model tracking enabled")
    
  # Track best loss for model saving
  best_loss = float('inf')
  
  # Training loop
  for epoch in range(1, epochs+1):
    logger.info(f"Starting epoch {epoch}/{epochs}")
    model.train()
    epoch_start_time = time.time()
    total_loss, sample_count = 0.0, 0
    progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}/{epochs}', ncols=100)
    
    batch_times = []
    forward_times = []
    backward_times = []
    
    for batch_idx, (queries, positive_docs, negative_docs) in enumerate(progress_bar):
      batch_start_time = time.time()
      
      # Move data to device
      queries = queries.to(device)
      positive_docs = positive_docs.to(device)
      negative_docs = negative_docs.to(device)
      
      if batch_idx == 0:  # Log tensor shapes for first batch
        logger.info(f"Batch {batch_idx} tensor shapes:")
        log_tensor_info(queries, "queries")
        log_tensor_info(positive_docs, "positive_docs")
        log_tensor_info(negative_docs, "negative_docs")
      
      # Forward pass
      forward_start = time.time()
      query_vectors = model.query_tower(queries)
      positive_doc_vectors = model.document_tower(positive_docs)
      negative_doc_vectors = model.document_tower(negative_docs)
      forward_time = time.time() - forward_start
      forward_times.append(forward_time)
      
      if batch_idx == 0:  # Log tensor shapes for first batch
        log_tensor_info(query_vectors, "query_vectors")
        log_tensor_info(positive_doc_vectors, "positive_doc_vectors")
        log_tensor_info(negative_doc_vectors, "negative_doc_vectors")
      
      # Compute loss using custom contrastive triplet loss
      loss = contrastive_triplet_loss(query_vectors, positive_doc_vectors, negative_doc_vectors, margin)
      
      # Backward pass and optimization
      backward_start = time.time()
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      backward_time = time.time() - backward_start
      backward_times.append(backward_time)
      
      # Calculate similarity scores for monitoring
      pos_similarity = F.cosine_similarity(query_vectors, positive_doc_vectors).mean().item()
      neg_similarity = F.cosine_similarity(query_vectors, negative_doc_vectors).mean().item()
      similarity_diff = pos_similarity - neg_similarity
      
      batch_loss = loss.item()
      total_loss += batch_loss * len(queries)
      sample_count += len(queries)
      
      # Calculate batch time
      batch_time = time.time() - batch_start_time
      batch_times.append(batch_time)
      
      # Log detailed metrics for first few batches
      if batch_idx < 3:
        logger.info(f"Batch {batch_idx} metrics:")
        logger.info(f"  Loss: {batch_loss:.6f}")
        logger.info(f"  Positive similarity: {pos_similarity:.6f}")
        logger.info(f"  Negative similarity: {neg_similarity:.6f}")
        logger.info(f"  Similarity difference: {similarity_diff:.6f}")
        logger.info(f"  Time: {batch_time:.4f}s (forward: {forward_time:.4f}s, backward: {backward_time:.4f}s)")
      
      # Update progress bar
      progress_bar.set_postfix({
        'loss': f'{batch_loss:.4f}',
        'pos_sim': f'{pos_similarity:.4f}',
        'neg_sim': f'{neg_similarity:.4f}',
        'diff': f'{similarity_diff:.4f}'
      })
      
      # Log metrics to wandb if enabled
      if use_wandb:
        wandb.log({
          'batch': epoch * len(data_loader) + batch_idx,
          'train/batch_loss': batch_loss,
          'train/pos_similarity': pos_similarity,
          'train/neg_similarity': neg_similarity,
          'train/similarity_diff': similarity_diff,
          'performance/batch_time': batch_time,
          'performance/forward_time': forward_time,
          'performance/backward_time': backward_time,
          'performance/samples_per_second': len(queries) / batch_time,
        })
      
        # Log gradient norms periodically (every 10 batches to avoid slowdown)
        if batch_idx % 10 == 0:
          # Compute gradient norms
          total_norm = 0
          for p in model.parameters():
            if p.grad is not None:
              param_norm = p.grad.detach().data.norm(2)
              total_norm += param_norm.item() ** 2
          total_norm = total_norm ** 0.5
          
          wandb.log({
            'batch': epoch * len(data_loader) + batch_idx,
            'gradients/total_norm': total_norm,
          })
    
    # Calculate epoch metrics
    epoch_loss = total_loss / sample_count
    epoch_time = time.time() - epoch_start_time
    avg_batch_time = sum(batch_times) / len(batch_times)
    avg_forward_time = sum(forward_times) / len(forward_times)
    avg_backward_time = sum(backward_times) / len(backward_times)
    
    logger.info(f"Epoch {epoch}/{epochs} completed in {epoch_time:.2f}s")
    logger.info(f"  Mean loss: {epoch_loss:.6f}")
    logger.info(f"  Average batch time: {avg_batch_time:.4f}s (forward: {avg_forward_time:.4f}s, backward: {avg_backward_time:.4f}s)")
    
    # Log epoch metrics to wandb if enabled
    if use_wandb:
      wandb.log({
        'epoch': epoch,
        'train/epoch_loss': epoch_loss,
        'train/epoch_time': epoch_time,
      })
      
    # Save best model
    if epoch_loss < best_loss:
      best_loss = epoch_loss
      logger.info(f"New best model with loss: {best_loss:.6f}")
      
      # Create timestamped filename
      timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
      checkpoint_name = f"two_tower_{timestamp}_epoch{epoch}.pt"
      checkpoint_path = os.path.join("checkpoints", checkpoint_name)
      
      # Save model checkpoint
      torch.save({
        'model': model.state_dict(),
        'vocab': dataset.vocab.string_to_index,
        'epoch': epoch,
        'loss': best_loss,
        'timestamp': timestamp
      }, checkpoint_path)
      
      # Also save as best_model.pt for easy reference
      best_model_path = os.path.join("checkpoints", "best_model.pt")
      torch.save({
        'model': model.state_dict(),
        'vocab': dataset.vocab.string_to_index,
        'epoch': epoch,
        'loss': best_loss,
        'timestamp': timestamp
      }, best_model_path)
      
      logger.info(f"Saved checkpoint to {checkpoint_path} and {best_model_path}")
      
      # Log as wandb artifact if enabled
      if use_wandb:
        artifact = wandb.Artifact(
            name=f"two-tower-model", 
            type="model",
            description=f"Two-Tower model from epoch {epoch} with loss {best_loss:.6f}"
        )
        artifact.add_file(checkpoint_path)
        
        # Log the artifact to W&B
        run = wandb.run
        run.log_artifact(artifact, aliases=["latest", "best"])
        logger.info(f"Logged model artifact to Weights & Biases")
  
  logger.info(f"Training completed. Best loss: {best_loss:.6f}")
  return model


#
# Simple retrieval demo
#
@torch.no_grad()
def retrieve(model, query_text, document_texts, vocab, top_k=5, device='cpu', max_length=64):
  logger.info(f"Retrieving top {top_k} documents for query: '{query_text}'")
  logger.info(f"Corpus size: {len(document_texts)} documents")
  
  # Apply the same truncate/pad logic that's used in dataset preparation
  def truncate_and_pad(sequence, max_len): 
    if len(sequence) < max_len:
      return sequence + [0] * (max_len - len(sequence))
    else:
      return sequence[:max_len]
  
  # Encode query and documents
  logger.info("Encoding query and documents")
  query_encoded = torch.tensor([truncate_and_pad(vocab.encode(query_text), max_length)], device=device)
  docs_encoded = torch.tensor([truncate_and_pad(vocab.encode(doc_text), max_length) for doc_text in document_texts], device=device)
  
  log_tensor_info(query_encoded, "query_encoded")
  log_tensor_info(docs_encoded, "docs_encoded")
  
  # Get embeddings
  logger.info("Computing embeddings")
  query_vector = model.query_tower(query_encoded)
  document_vectors = model.document_tower(docs_encoded)
  
  log_tensor_info(query_vector, "query_vector")
  log_tensor_info(document_vectors, "document_vectors")
  
  # Compute similarities and get top-k
  logger.info("Computing similarities and ranking documents")
  similarity_scores = (query_vector @ document_vectors.t()).squeeze(0)   # (num_docs,)
  top_indices = similarity_scores.argsort(descending=True)[:top_k]
  
  log_tensor_info(similarity_scores, "similarity_scores")
  log_tensor_info(top_indices, "top_indices")
  
  # Return results
  results = [(document_texts[idx], similarity_scores[idx].item()) for idx in top_indices]
  logger.info(f"Top {len(results)} results:")
  for i, (doc_text, score) in enumerate(results):
    logger.info(f"  {i+1}. Score: {score:.4f}, Document: '{doc_text[:50]}...'")
  
  return results


#
# CLI
#
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data', default='pairs.parquet', help='pairs.parquet or pairs.tsv')
  parser.add_argument('--epochs', type=int, default=3)
  parser.add_argument('--batch_size', type=int, default=256)
  parser.add_argument('--learning_rate', type=float, default=1e-3)
  parser.add_argument('--device', default='cpu')
  parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
  parser.add_argument('--wandb_project', default=os.environ.get('WANDB_PROJECT', 'two-tower'), help='W&B project name')
  parser.add_argument('--wandb_entity', default=os.environ.get('WANDB_ENTITY', None), help='W&B entity (username or org)')
  parser.add_argument('--wandb_run_name', default=None, help='W&B run name')
  parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', 
                      help='Logging level')
  args = parser.parse_args()

  # Set log level
  logger.setLevel(getattr(logging, args.log_level))
  logger.info(f"Log level set to {args.log_level}")
  
  # Log all arguments
  logger.info(f"Command-line arguments: {args}")
  
  # Log system info
  import platform
  import sys
  logger.info(f"Python version: {platform.python_version()}")
  logger.info(f"PyTorch version: {torch.__version__}")
  logger.info(f"System: {platform.system()} {platform.release()}")
  if torch.cuda.is_available():
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

  # Initialize wandb if enabled
  if args.wandb:
    # API key will be automatically loaded from WANDB_API_KEY in .env
    wandb.init(
      project=args.wandb_project,
      entity=args.wandb_entity,
      name=args.wandb_run_name,
      config={
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'device': args.device,
        'data_path': args.data,
      }
    )
    logger.info(f"Weights & Biases initialized: {wandb.run.name}")

  # Load dataset
  logger.info(f"Loading dataset from {args.data}")
  dataset = Triplets(args.data)
  
  # Create model
  logger.info("Creating model")
  model = TwoTower(len(dataset.vocab))
  
  # Log dataset info to wandb if enabled
  if args.wandb:
    # Log dataset and model architecture details
    wandb.config.update({
      'vocab_size': len(dataset.vocab),
      'triplets_count': len(dataset.triplets),
      'model': {
        'embedding_dim': 64,  # Default value, update if changed
        'hidden_dim': 128,    # Default value, update if changed
        'query_tower_params': sum(p.numel() for p in model.query_tower.parameters()),
        'doc_tower_params': sum(p.numel() for p in model.document_tower.parameters()),
        'total_params': sum(p.numel() for p in model.parameters()),
      },
      'hardware': {
        'device': args.device,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
      },
    })
    
    # Log example data samples
    example_query = dataset.query_texts[0] if dataset.query_texts else ""
    example_pos_doc = dataset.positive_doc_texts[0] if dataset.positive_doc_texts else ""
    example_neg_doc = dataset.negative_doc_texts[0] if dataset.negative_doc_texts else ""
    
    wandb.log({
      "examples/query": wandb.Html(f"<p>{example_query}</p>"),
      "examples/positive_doc": wandb.Html(f"<p>{example_pos_doc[:200]}...</p>"),
      "examples/negative_doc": wandb.Html(f"<p>{example_neg_doc[:200]}...</p>"),
    })
  
  # Train model
  logger.info("Starting training")
  model = train(
    model, 
    dataset, 
    epochs=args.epochs, 
    batch_size=args.batch_size, 
    learning_rate=args.learning_rate,
    device=args.device,
    use_wandb=args.wandb
  )
  
  # Save final model
  logger.info("Saving final model")
  torch.save({
    'model': model.state_dict(),
    'vocab': dataset.vocab.string_to_index
  }, 'model.pt')
  
  logger.info('Final model saved to model.pt')
  logger.info('Best model saved to best_model.pt')

  # Quick sanity check
  logger.info("Performing retrieval demo")
  sample_size = min(20, len(dataset.positive_doc_texts))
  document_indices = random.sample(range(len(dataset.positive_doc_texts)), sample_size)
  sample_documents = [dataset.positive_doc_texts[idx] for idx in document_indices]
  
  logger.info(f"Sample corpus size: {len(sample_documents)} documents")
  logger.info(f"Sample document: '{sample_documents[0][:50]}...'")
  
  query_text = input('Type a query: ')
  logger.info(f"User query: '{query_text}'")
  
  results = retrieve(model, query_text, sample_documents, dataset.vocab)
  
  print("\nRetrieval results:")
  for i, (document_text, score) in enumerate(results):
    print(f"  {i+1}. {score:+.3f} | {document_text[:80]}...")
    
  # Log retrieval results to wandb if enabled
  if args.wandb and results:
    retrieval_table = wandb.Table(columns=["Rank", "Score", "Document"])
    for i, (document_text, score) in enumerate(results):
      retrieval_table.add_data(i+1, score, document_text[:200] + "...")
    
    wandb.log({"retrieval_results": retrieval_table})

  # Close wandb run if enabled
  if args.wandb:
    logger.info("Finishing wandb run")
    wandb.finish()


if __name__ == '__main__':
  main()

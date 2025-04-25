import os
import logging
from typing import List, Optional
import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import chromadb
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
import os.path
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
MODEL_REPO_URL = os.getenv("MODEL_REPO_URL", "https://huggingface.co/azuremis/twotower-char-emb")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
PORT = int(os.getenv("PORT", "8080"))

# Initialize FastAPI
app = FastAPI(title="Two Tower Inference API")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Models
class Query(BaseModel):
    text: str
    top_k: Optional[int] = 5

class EmbeddingRequest(BaseModel):
    texts: List[str]

class SearchResult(BaseModel):
    id: str
    text: str
    score: float

# Custom model wrapper to handle your two-tower model
class TwoTowerModel:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the PyTorch model
        self.model = torch.load(model_path, map_location=self.device)
        logger.info(f"Model loaded with keys: {list(self.model.keys() if isinstance(self.model, dict) else ['not a dict'])}")

        # If it's a state_dict, we need to know the model architecture
        if isinstance(self.model, dict):
            # This is a checkpoint with state_dict
            if "model" in self.model:
                self.model = self.model["model"]
            if "tokenizer" in self.model:
                self.tokenizer = self.model["tokenizer"]
            else:
                # Fall back to a simple whitespace tokenizer
                self.tokenizer = lambda text: text.split()
        
        # Move model to the appropriate device
        if hasattr(self.model, 'to'):
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

    def encode(self, texts, batch_size=32):
        """Encode texts to embeddings"""
        if not isinstance(texts, list):
            texts = [texts]
        
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                try:
                    # Try to use the model's encode method if it exists
                    if hasattr(self.model, 'encode'):
                        batch_embeddings = self.model.encode(batch)
                    # Otherwise, assume it's a query tower that takes tokenized input
                    elif hasattr(self.model, 'forward'):
                        # Try to tokenize if there's a tokenizer
                        if hasattr(self, 'tokenizer'):
                            tokenized = [self.tokenizer(text) for text in batch]
                            inputs = torch.tensor(tokenized).to(self.device)
                        else:
                            # Just convert text to character indices as fallback
                            inputs = [[ord(c) % 256 for c in text] for text in batch]
                            max_len = max(len(seq) for seq in inputs)
                            inputs = [seq + [0] * (max_len - len(seq)) for seq in inputs]
                            inputs = torch.tensor(inputs).to(self.device)
                        
                        batch_embeddings = self.model(inputs).cpu().numpy()
                    else:
                        raise ValueError("Unknown model format, can't encode texts")
                except Exception as e:
                    logger.error(f"Error encoding batch: {e}")
                    # Fallback to using SentenceTransformer
                    logger.info("Falling back to SentenceTransformer")
                    encoder = SentenceTransformer("all-MiniLM-L6-v2")
                    batch_embeddings = encoder.encode(batch)
                
                embeddings.extend(batch_embeddings)
        
        # Ensure embeddings are numpy arrays
        if not isinstance(embeddings, np.ndarray):
            try:
                embeddings = np.array(embeddings)
            except:
                # If conversion fails, each embedding might be complex objects
                # Try to convert them individually
                for i in range(len(embeddings)):
                    if hasattr(embeddings[i], 'cpu'):
                        embeddings[i] = embeddings[i].cpu().numpy()
        
        return embeddings

# Global variables
model = None
chroma_client = None
collection = None

@app.on_event("startup")
async def startup_event():
    global model, chroma_client, collection
    
    # Download model from Hugging Face Hub
    logger.info(f"Downloading model from {MODEL_REPO_URL}")
    try:
        model_path = snapshot_download(repo_id=MODEL_REPO_URL)
        model_file = os.path.join(model_path, "best_model.pt")
        
        if os.path.exists(model_file):
            logger.info(f"Loading model from {model_file}")
            model = TwoTowerModel(model_file)
        else:
            # Fallback to SentenceTransformer if best_model.pt is not found
            logger.info("best_model.pt not found, falling back to SentenceTransformer")
            model = SentenceTransformer(model_path)
            
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Falling back to default SentenceTransformer model")
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Fallback model loaded successfully")
        except Exception as e2:
            logger.error(f"Failed to load fallback model: {e2}")
            raise RuntimeError(f"Failed to load any model: {e2}")
    
    # Connect to ChromaDB
    logger.info(f"Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}")
    try:
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        # Check if the collection exists, if not create it
        try:
            collection = chroma_client.get_collection("embeddings")
            logger.info("Connected to existing ChromaDB collection 'embeddings'")
        except:
            collection = chroma_client.create_collection("embeddings")
            logger.info("Created new ChromaDB collection 'embeddings'")
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {e}")
        # Don't raise error here to allow API to start even if ChromaDB is not available yet

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    health_status = {
        "status": "healthy",
        "model_loaded": model is not None,
        "chroma_connected": chroma_client is not None and collection is not None
    }
    return health_status

@app.post("/embed")
async def embed(request: EmbeddingRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        embeddings = model.encode(request.texts)
        return {"success": True, "count": len(embeddings)}
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

@app.post("/search")
async def search(query: Query):
    if model is None or collection is None:
        raise HTTPException(status_code=503, detail="Model or ChromaDB not available")
    
    try:
        # Encode query
        query_embedding = model.encode(query.text)
        
        # Ensure embedding is in the correct format for Chroma
        # Handle nested arrays if the model returns them
        import numpy as np
        
        # Debug log the shape and type
        logger.info(f"Query embedding type: {type(query_embedding)}")
        if isinstance(query_embedding, list) and len(query_embedding) > 0:
            logger.info(f"First element type: {type(query_embedding[0])}")
            # If we get a list of lists but only have one query, take the first element
            if isinstance(query_embedding[0], list):
                query_embedding = query_embedding[0]
        elif isinstance(query_embedding, np.ndarray):
            # If it's a 2D array but we only have one query, take the first row
            if len(query_embedding.shape) > 1 and query_embedding.shape[0] == 1:
                query_embedding = query_embedding[0]
                
        # Final conversion to list
        if isinstance(query_embedding, np.ndarray):
            query_embedding_list = query_embedding.tolist()
        else:
            query_embedding_list = list(query_embedding)
        
        # Search in Chroma
        results = collection.query(
            query_embeddings=[query_embedding_list],
            n_results=query.top_k
        )
        
        # Format results
        search_results = []
        if results["ids"] and results["distances"]:
            for i, (doc_id, score) in enumerate(zip(results["ids"][0], results["distances"][0])):
                metadata = results["metadatas"][0][i] if "metadatas" in results else {}
                text = metadata.get("text", "No text available")
                search_results.append(SearchResult(id=doc_id, text=text, score=float(score)))
        
        return {"results": search_results}
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.post("/add")
async def add_documents(request: EmbeddingRequest):
    if model is None or collection is None:
        raise HTTPException(status_code=503, detail="Model or ChromaDB not available")
    
    try:
        # Generate IDs
        ids = [f"doc_{i}" for i in range(len(request.texts))]
        
        # Generate embeddings
        embeddings = model.encode(request.texts)
        
        # Debug log the shape and type
        logger.info(f"Embeddings type: {type(embeddings)}")
        if isinstance(embeddings, list) and len(embeddings) > 0:
            logger.info(f"First embedding type: {type(embeddings[0])}")
        
        # Process embeddings to make them suitable for Chroma
        import numpy as np
        embedding_lists = []
        
        # Handle different types of embeddings
        if isinstance(embeddings, np.ndarray):
            # If we have a 2D numpy array (the normal case)
            if len(embeddings.shape) == 2:
                embedding_lists = embeddings.tolist()
            # If we have a 3D array (batch of 2D embeddings)
            elif len(embeddings.shape) == 3:
                embedding_lists = [row[0].tolist() for row in embeddings]
            else:
                embedding_lists = [embeddings.tolist()]
        elif isinstance(embeddings, list):
            # If it's already a list
            if all(isinstance(emb, np.ndarray) for emb in embeddings):
                # List of numpy arrays
                embedding_lists = [emb.tolist() for emb in embeddings]
            elif all(isinstance(emb, list) for emb in embeddings):
                # List of lists, check if they're nested too deeply
                if embeddings and isinstance(embeddings[0], list) and embeddings[0] and isinstance(embeddings[0][0], list):
                    # It's a list of lists of lists - flatten by one level
                    embedding_lists = [emb[0] for emb in embeddings]
                else:
                    embedding_lists = embeddings
            else:
                # Not sure what we have - just try to make it work
                embedding_lists = [list(emb) if not isinstance(emb, list) else emb for emb in embeddings]
        
        logger.info(f"Final embeddings shape: {len(embedding_lists)} items of dimension {len(embedding_lists[0]) if embedding_lists else 'unknown'}")
        
        # Add to Chroma
        metadatas = [{"text": text} for text in request.texts]
        collection.add(
            ids=ids,
            embeddings=embedding_lists,
            metadatas=metadatas
        )
        
        return {"success": True, "added": len(request.texts), "ids": ids}
    except Exception as e:
        logger.error(f"Add documents error: {e}")
        raise HTTPException(status_code=500, detail=f"Add documents error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False) 
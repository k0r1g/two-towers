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
        model = SentenceTransformer(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")
    
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
        
        # Search in Chroma
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
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
        
        # Add to Chroma
        metadatas = [{"text": text} for text in request.texts]
        collection.add(
            ids=ids,
            embeddings=[embedding.tolist() for embedding in embeddings],
            metadatas=metadatas
        )
        
        return {"success": True, "added": len(request.texts), "ids": ids}
    except Exception as e:
        logger.error(f"Add documents error: {e}")
        raise HTTPException(status_code=500, detail=f"Add documents error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False) 
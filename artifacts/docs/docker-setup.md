# Docker Setup for Two-Tower Model Inference

This document describes the Docker setup for deploying the Two-Tower model inference service with ChromaDB.

## Overview

The setup consists of two Docker services:

1. **ChromaDB**: A vector database for storing and querying embeddings
2. **Inference API**: A FastAPI service that loads our Two-Tower model from Hugging Face and provides endpoints for embedding generation and vector search

```
┌─────────────┐     ┌─────────────┐
│             │     │             │
│  Inference  │◄────►   ChromaDB  │
│   Service   │     │   Service   │
│             │     │             │
└─────┬───────┘     └─────────────┘
      │                    ▲
      │                    │
      │                    │
      ▼                    │
┌─────────────┐            │
│             │            │
│   Client    │────────────┘
│  Requests   │
│             │
└─────────────┘
```

## File Structure

```
two-towers/
├─ docker-compose.yml                # Docker Compose configuration
├─ data/
│  └─ chroma_data/                  # Persistent storage for ChromaDB
└─ inference/
   └─ docker/                       # Docker files for inference service
      ├─ Dockerfile                 # Container configuration
      ├─ requirements.txt           # Python dependencies
      ├─ app.py                     # FastAPI application
      ├─ .dockerignore              # Files to exclude from build
      └─ static/                    # Static files for web UI
         └─ index.html              # Demo web interface
```

## Quick Start

To start the services:

```bash
cd two-towers
sudo docker compose up -d
```

The services will be available at:
- ChromaDB: http://localhost:8000
- Inference API: http://localhost:8080

To stop the services:

```bash
sudo docker compose down
```

## Services Details

### ChromaDB Service

The ChromaDB service uses the official ChromaDB Docker image to provide a vector database that persists data to disk.

```yaml
chroma:
  image: chromadb/chroma:latest
  volumes:
    - ./data/chroma_data:/chroma/.chroma
  environment:
    - IS_PERSISTENT=TRUE
  ports: ["8000:8000"]
```

- **Data Persistence**: Data is stored in the `./data/chroma_data` directory
- **Port**: The service is accessible on port 8000

### Inference Service

The Inference service is a custom Python application built with FastAPI that:

1. Loads our Two-Tower model from Hugging Face
2. Provides an API for generating embeddings
3. Connects to ChromaDB for vector storage and search
4. Serves a web-based demo interface

```yaml
inference:
  build: ./inference/docker
  environment:
    - MODEL_REPO_URL=Azuremis/mlx7-two-tower-retrieval
    - CHROMA_HOST=chroma
  depends_on: [chroma]
  ports: ["8080:8080"]
```

- **Model**: The service loads the model from the specified Hugging Face repository
- **Port**: The API is accessible on port 8080
- **Dependencies**: Requires the ChromaDB service to be running

## Custom Model Loading

The Inference service can handle different types of Two-Tower models:

### Custom Model Wrapper

The service includes a custom `TwoTowerModel` wrapper class that handles loading and inference with PyTorch model files:

```python
class TwoTowerModel:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the PyTorch model
        self.model = torch.load(model_path, map_location=self.device)
        
        # Handle different model structures
        if isinstance(self.model, dict):
            # This is a checkpoint with state_dict
            if "model" in self.model:
                self.model = self.model["model"]
            if "tokenizer" in self.model:
                self.tokenizer = self.model["tokenizer"]
            else:
                # Fall back to a simple whitespace tokenizer
                self.tokenizer = lambda text: text.split()
        
        # Set to evaluation mode
        if hasattr(self.model, 'to'):
            self.model.to(self.device)
            self.model.eval()
```

### Model Loading Strategy

The service employs a multi-tiered model loading strategy:

1. **Primary**: Tries to load the PyTorch model file (`best_model.pt`) from the specified Hugging Face repository
2. **Secondary**: Falls back to loading the repository as a SentenceTransformer model if the PyTorch file isn't found
3. **Fallback**: As a last resort, loads a standard SentenceTransformer model (`all-MiniLM-L6-v2`)

```python
# Download model from Hugging Face Hub
try:
    model_path = snapshot_download(repo_id=MODEL_REPO_URL)
    model_file = os.path.join(model_path, "best_model.pt")
    
    if os.path.exists(model_file):
        # Load custom PyTorch model
        model = TwoTowerModel(model_file)
    else:
        # Try to load as SentenceTransformer
        model = SentenceTransformer(model_path)
except Exception as e:
    # Fall back to default SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
```

### Embedding Generation

The custom model wrapper generates embeddings by trying different approaches in order:

1. Call the model's `encode` method if it exists
2. Call the model's `forward` method with tokenized input
3. Fall back to a standard SentenceTransformer embedding

```python
def encode(self, texts, batch_size=32):
    """Encode texts to embeddings"""
    if not isinstance(texts, list):
        texts = [texts]
    
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            try:
                # Try different methods in order
                if hasattr(self.model, 'encode'):
                    batch_embeddings = self.model.encode(batch)
                elif hasattr(self.model, 'forward'):
                    # Tokenize and encode
                    # ...
                else:
                    raise ValueError("Unknown model format")
            except Exception as e:
                # Fall back to SentenceTransformer
                encoder = SentenceTransformer("all-MiniLM-L6-v2")
                batch_embeddings = encoder.encode(batch)
```

## Handling Different Embedding Formats

The service is designed to handle various embedding formats that might be returned by different models:

### For Search Queries

```python
# Ensure embedding is in the correct format for Chroma
logger.info(f"Query embedding type: {type(query_embedding)}")

# Handle list of lists format
if isinstance(query_embedding, list) and len(query_embedding) > 0:
    if isinstance(query_embedding[0], list):
        query_embedding = query_embedding[0]
# Handle numpy arrays
elif isinstance(query_embedding, np.ndarray):
    if len(query_embedding.shape) > 1 and query_embedding.shape[0] == 1:
        query_embedding = query_embedding[0]
```

### For Batch Embeddings

The service can handle various embedding output formats:

- 2D and 3D numpy arrays
- Lists of numpy arrays
- Nested lists
- Mixed formats

```python
# Process embeddings to make them suitable for Chroma
if isinstance(embeddings, np.ndarray):
    # Handle different numpy array dimensions
    if len(embeddings.shape) == 2:
        embedding_lists = embeddings.tolist()
    elif len(embeddings.shape) == 3:
        embedding_lists = [row[0].tolist() for row in embeddings]
    else:
        embedding_lists = [embeddings.tolist()]
elif isinstance(embeddings, list):
    # Handle different list structures
    # ...
```

## API Endpoints

The Inference service provides the following endpoints:

### `GET /`

Serves the web-based demo interface.

### `GET /health`

Health check endpoint that returns information about the service status:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "chroma_connected": true
}
```

### `POST /embed`

Generate embeddings for a list of texts:

**Request:**
```json
{
  "texts": ["sample text 1", "sample text 2"]
}
```

**Response:**
```json
{
  "success": true,
  "count": 2
}
```

### `POST /search`

Search for similar texts in the vector database:

**Request:**
```json
{
  "text": "query text",
  "top_k": 5
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "doc_1",
      "text": "similar text 1",
      "score": 0.85
    },
    {
      "id": "doc_2",
      "text": "similar text 2",
      "score": 0.75
    }
  ]
}
```

### `POST /add`

Add texts to the vector database:

**Request:**
```json
{
  "texts": ["text to add 1", "text to add 2"]
}
```

**Response:**
```json
{
  "success": true,
  "added": 2,
  "ids": ["doc_0", "doc_1"]
}
```

## Environment Variables

### ChromaDB Service

- `IS_PERSISTENT`: Set to TRUE to enable data persistence

### Inference Service

- `MODEL_REPO_URL`: The Hugging Face repository ID (e.g., `Azuremis/mlx7-two-tower-retrieval`)
- `CHROMA_HOST`: The hostname of the ChromaDB service (default: chroma)
- `CHROMA_PORT`: The port of the ChromaDB service (default: 8000)
- `PORT`: The port for the Inference API (default: 8080)

## Web Interface

The service includes a simple web interface at the root URL (`/`) that allows you to:

1. Add documents to the vector database
2. Search for similar documents
3. Generate embeddings for text input

## Using Your Own Model

To use your own Two-Tower model:

1. Upload your model to Hugging Face Hub
   - Export your model using `torch.save()` and save as `best_model.pt`
   - Upload to your Hugging Face repository

2. Update the MODEL_REPO_URL environment variable in docker-compose.yml:
   ```yaml
   environment:
     - MODEL_REPO_URL=your-username/your-model-repo
   ```

3. Restart the services:
   ```bash
   sudo docker compose down
   sudo docker compose up -d
   ```

## Development

To modify the Inference service:

1. Update files in the `inference/docker` directory
2. Rebuild the Docker image: `sudo docker compose build inference`
3. Restart the service: `sudo docker compose up -d inference`

## Troubleshooting

If you encounter issues:

1. Check the logs: `sudo docker compose logs`
2. Verify ChromaDB is running: `curl http://localhost:8000/api/v1/heartbeat`
3. Check the Inference API health: `curl http://localhost:8080/health`
4. Restart the services: `sudo docker compose restart`

### Common Issues

1. **Model Loading Errors**:
   - Make sure your model file is named `best_model.pt` in the Hugging Face repository
   - Check that your model format is compatible (PyTorch model or state dictionary)
   - Verify that the model repository is publicly accessible

2. **Embedding Format Errors**:
   - The system will try to handle various embedding formats, but very custom formats might need code adjustments
   - Check the logs for details about the shape and format of embeddings

3. **Permission Issues with Docker**:
   - If you encounter permission denied errors when running docker commands, add your user to the docker group:
     ```bash
     sudo usermod -aG docker $USER
     ```
   - Log out and log back in for the changes to take effect, or run with sudo until then 
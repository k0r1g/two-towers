# Docker Setup Documentation

## System Architecture

The system consists of two main services:

1. **ChromaDB**: A vector database for storing and querying embeddings
2. **Inference API**: A FastAPI service that loads a model from Hugging Face and provides endpoints for embedding generation and vector search

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

## Prerequisites

- Docker
- Docker Compose

## Services Details

### ChromaDB Service

The ChromaDB service uses the official ChromaDB Docker image to provide a vector database that persists data to disk.

```yaml
chroma:
  image: chromadb/chroma:latest
  volumes:
    - ./chroma_data:/chroma/.chroma
  environment:
    - IS_PERSISTENT=TRUE
  ports: ["8000:8000"]
```

- **Data Persistence**: Data is stored in the `./chroma_data` directory
- **Port**: The service is accessible on port 8000

### Inference Service

The Inference service is a custom Python application built with FastAPI that:

1. Loads a model from Hugging Face
2. Provides an API for generating embeddings
3. Connects to ChromaDB for vector storage and search

```yaml
inference:
  build: ./docker/inference
  environment:
    - MODEL_REPO_URL=https://huggingface.co/azuremis/twotower-char-emb
    - CHROMA_HOST=chroma
  depends_on: [chroma]
  ports: ["8080:8080"]
```

- **Model**: The service loads a model from the specified Hugging Face repository
- **Port**: The API is accessible on port 8080
- **Dependencies**: Requires the ChromaDB service to be running

## API Endpoints

The Inference service provides the following endpoints:

### `GET /`

Root endpoint that returns a simple message confirming the API is running.

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

## Docker Compose Commands

### Start the Services

```bash
sudo docker compose up -d
```

### View Logs

```bash
sudo docker compose logs -f
```

### Stop the Services

```bash
sudo docker compose down
```

### Rebuild the Services

```bash
sudo docker compose build
```

## Environment Variables

### ChromaDB Service

- `IS_PERSISTENT`: Set to TRUE to enable data persistence

### Inference Service

- `MODEL_REPO_URL`: The Hugging Face repository URL for the model
- `CHROMA_HOST`: The hostname of the ChromaDB service
- `CHROMA_PORT`: The port of the ChromaDB service (default: 8000)
- `PORT`: The port for the Inference API (default: 8080)

## Storage

The ChromaDB service stores data in the `./chroma_data` directory. This directory is mounted as a volume in the ChromaDB container, ensuring that data persists even if the container is restarted or removed.

## Development

### Dockerfile

The Inference service uses a Python-based Dockerfile:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables with defaults
ENV MODEL_REPO_URL=https://huggingface.co/azuremis/twotower-char-emb
ENV CHROMA_HOST=chroma
ENV CHROMA_PORT=8000
ENV PORT=8080

# Expose the port the app runs on
EXPOSE ${PORT}

# Command to run the application
CMD ["python", "app.py"]
```

### Python Dependencies

The Inference service requires the following Python packages:

```
torch>=2.0.0
transformers>=4.28.0
fastapi>=0.95.0
uvicorn>=0.21.1
chromadb>=0.4.0
huggingface_hub>=0.14.1
pydantic>=1.10.7
sentence-transformers>=2.2.2
numpy>=1.24.3
``` 
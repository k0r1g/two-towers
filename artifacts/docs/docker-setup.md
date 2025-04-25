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
    - MODEL_REPO_URL=https://huggingface.co/azuremis/twotower-char-emb
    - CHROMA_HOST=chroma
  depends_on: [chroma]
  ports: ["8080:8080"]
```

- **Model**: The service loads the model from the specified Hugging Face repository
- **Port**: The API is accessible on port 8080
- **Dependencies**: Requires the ChromaDB service to be running

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

- `MODEL_REPO_URL`: The Hugging Face repository URL for the model (default: https://huggingface.co/azuremis/twotower-char-emb)
- `CHROMA_HOST`: The hostname of the ChromaDB service (default: chroma)
- `CHROMA_PORT`: The port of the ChromaDB service (default: 8000)
- `PORT`: The port for the Inference API (default: 8080)

## Web Interface

The service includes a simple web interface at the root URL (`/`) that allows you to:

1. Add documents to the vector database
2. Search for similar documents
3. Generate embeddings for text input

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
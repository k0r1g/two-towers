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
# Simple In-Memory RAG (No Vector DB)

This project demonstrates a minimal Retrieval-Augmented Generation (RAG) pipeline using:

- Local OpenAI-compatible embeddings endpoint
- Local OpenAI-compatible chat model endpoint
- In-memory Python list as the vector index (no Chroma/FAISS/Pinecone)

## What This Program Does

1. Loads documents from `get_doc()`
2. Splits each document into chunks
3. Creates embeddings for each chunk
4. Stores chunk + embedding rows in memory
5. For each query:
   - Embeds the query
   - Scores similarity against all chunk embeddings
   - Retrieves top-k chunks
   - Sends retrieved context to LLM for final answer

## File

- `simple-rag.py`: Full RAG pipeline in one file

## Requirements

Install dependencies from `requirements.txt`:

```powershell
pip install -r requirements.txt
```

## Local Model Setup

Your local server is expected at:

- `http://localhost:1234/v1`

The code uses:

- Chat model: `nvidia/nemotron-3-nano-4b`
- Embedding model (default): `text-embedding-nomic-embed-text-v1.5`

If your local embedding model name is different, set:

```powershell
$env:LOCAL_EMBED_MODEL="your-embedding-model-name"
```

## Run

```powershell
python simple-rag.py
```

## Output Behavior

The sample `main()` asks:

1. `What is the capital of Germany?`  
Expected: model answers `Berlin` from retrieved context.

2. `What is the capital of India?`  
Expected: model says it does not know (because India is not in provided docs).

## How Retrieval Works (No Store)

The in-memory index shape is:

```python
[
  {
    "document_id": "doc2",
    "chunk_text": "The capital of Germany is Berlin.",
    "embedding": [ ...vector... ]
  },
  ...
]
```

Retrieval computes cosine similarity between query embedding and each row embedding, then takes top-k highest scores.

## Common Errors and Fixes

1. `400: 'input' field must be a string or an array of strings`  
Cause: provider rejects tokenized input.  
Fix: already handled in code with:
- `check_embedding_ctx_length=False`
- `tiktoken_enabled=False`

2. Connection error to `localhost:1234`  
Cause: local model server not running.  
Fix: start LM Studio/Ollama-compatible server first.

3. Model not found  
Cause: model name in code/env does not match loaded model.  
Fix: set `LOCAL_EMBED_MODEL` to exact loaded embedding model name.

## Notes

- This is intentionally simple and great for learning/debugging.
- For larger datasets, move from in-memory rows to a persistent vector store.

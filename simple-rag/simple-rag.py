import os
from typing import Any
import math

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_doc() -> list[dict[str, str]]:
    """
    Return our source knowledge base.

    Each item has:
    - id: stable identifier for traceability/debugging
    - text: raw content that can be chunked and embedded
    """
    return [
        {"text": "The capital of France is Paris." , "id": "doc1"},
        {"text": "The capital of Germany is Berlin.", "id": "doc2"},
        {"text": "The capital of Italy is Rome.", "id": "doc3"},
    ]


def embed_docs(docs: list[dict[str, str]]) -> list[dict[str, Any]]:
    """
    Build an in-memory embedding index from the input documents.

    Output shape (one row per CHUNK, not per original document):
    [
        {
            "document_id": "doc1",
            "chunk_text": "The capital of France is Paris.",
            "embedding": [0.01, -0.12, ...]
        },
        ...
    ]
    """
    # Splitter controls retrieval granularity.
    # Smaller chunks can improve precision but may lose context.
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

    # Embeddings client points to local OpenAI-compatible endpoint.
    embedding_model = local_embeddings()

    # This acts as our "vector store" in memory.
    rows: list[dict[str, Any]] = []

    # Process each source document independently.
    for original_doc in docs:
        # Chunk a single document's text into smaller strings.
        chunks = splitter.split_text(original_doc["text"])  # list[str]

        # Embed all chunks in one request for efficiency.
        # vectors[i] corresponds to chunks[i].
        vectors = embedding_model.embed_documents(chunks)  # list[list[float]]

        # zip keeps chunk text aligned with its exact embedding vector.
        for chunk, vector in zip(chunks, vectors):
            rows.append({
                # Preserve source document id to explain retrieval results later.
                "document_id": original_doc["id"],
                "chunk_text": chunk,
                "embedding": vector,
            })

    return rows


def local_embeddings() -> OpenAIEmbeddings:
    """
    Create embeddings client for a local OpenAI-compatible server.

    LOCAL_EMBED_MODEL lets you switch the embedding model at runtime:
    PowerShell example:
        $env:LOCAL_EMBED_MODEL="text-embedding-nomic-embed-text-v1.5"
    """
    return OpenAIEmbeddings(
        base_url="http://localhost:1234/v1",
        model=os.getenv("LOCAL_EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5"),
        api_key="fake-api-key",
        # Force raw string inputs. Some local providers reject token-id arrays.
        check_embedding_ctx_length=False,
        tiktoken_enabled=False,
    )

def local_llm() -> ChatOpenAI:
    """Create chat model client for local OpenAI-compatible server."""
    return ChatOpenAI(
        base_url="http://localhost:1234/v1",
        model="nvidia/nemotron-3-nano-4b",
        api_key="fake-api-key",
        temperature=0,
    )

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    # Dot product captures directional alignment.
    dot = sum(x * y for x, y in zip(a, b))
    # Norms scale vectors to unit length for cosine distance.
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def retrieve_top_k(rows: list[dict[str, Any]], query: str, k: int = 3) -> list[dict[str, Any]]:
    """
    Retrieve top-k most relevant chunks for a query.

    Steps:
    1. Embed the query
    2. Score query vs every chunk (cosine similarity)
    3. Sort by score descending
    """
    # Query and docs must use the same embedding model space.
    q_vec = local_embeddings().embed_query(query)

    scored = []
    for row in rows:
        score = cosine_similarity(q_vec, row["embedding"])
        scored.append({**row, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]


def answer_query(rows: list[dict[str, Any]], query: str, k: int = 3) -> str:
    """
    Run retrieval-augmented generation:
    - retrieve relevant chunks
    - build grounded prompt
    - ask LLM to answer using only retrieved context
    """
    top_chunks = retrieve_top_k(rows, query, k=k)
    # Concatenate top chunks into one context block.
    context = "\n".join([c["chunk_text"] for c in top_chunks])

    # Prompt constrains the model to avoid hallucinating beyond context.
    prompt = f"""Use only the context below.

            Context:
            {context}

            Question: {query}
            If answer is not in context, say "I don't know."
            """
    return local_llm().invoke(prompt).content


def main() -> None:
    # 1) Build in-memory vector index once.
    docs = get_doc()
    rows = embed_docs(docs)

    # 2) Ask questions against the in-memory index.
    answer = answer_query(rows, "What is the capital of Germany?")
    print(answer)

    # This should typically return "I don't know." because answer is absent in context.
    answer = answer_query(rows, "What is the capital of India?")
    print(answer)

if __name__ == "__main__":
    main()

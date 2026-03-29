"""Model client factories.

This module centralizes all model endpoint wiring so indexing/query modules do
not need to know connection details.
"""

import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def local_embeddings() -> OpenAIEmbeddings:
    """Create embeddings client for a local OpenAI-compatible endpoint.

    Uses env override:
    - LOCAL_EMBED_MODEL

    Notes:
    - `check_embedding_ctx_length=False` and `tiktoken_enabled=False` help with
      certain local providers that do not fully mimic OpenAI tokenizer behavior.
    """
    return OpenAIEmbeddings(
        base_url="http://localhost:1234/v1",
        model=os.getenv("LOCAL_EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5"),
        api_key="fake-api-key",
        check_embedding_ctx_length=False,
        tiktoken_enabled=False,
    )


def local_llm() -> ChatOpenAI:
    """Create chat model client for local OpenAI-compatible endpoint."""
    return ChatOpenAI(
        base_url="http://localhost:1234/v1",
        model="nvidia/nemotron-3-nano-4b",
        api_key="fake-api-key",
        temperature=0,
    )

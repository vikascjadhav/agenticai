"""Vector store indexing with incremental update support.

Responsibilities:
- Open/create Chroma collection.
- Compare configured sources to manifest fingerprints.
- Re-index only changed sources.
- Remove stale sources no longer configured.
- Emit structured indexing logs.
"""

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_app.clients import local_embeddings
from rag_app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    MANIFEST_PATH,
    PERSIST_DIRECTORY,
)
from rag_app.logging_utils import log_event
from rag_app.manifest import load_manifest, save_manifest, source_fingerprint, source_key
from rag_app.pdf_loader import read_pdf_pages


def load_vector_store(sources: list[dict]) -> Chroma:
    """Create/update vector store using incremental source-level indexing.

    Flow summary:
    1) Open persistent Chroma collection.
    2) Remove vectors for sources deleted from config.
    3) For each configured source:
       - skip if fingerprint unchanged
       - otherwise read pages, split to chunks, replace source vectors
    4) Save updated manifest and return ready-to-query vector store.
    """
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=local_embeddings(),
        persist_directory=PERSIST_DIRECTORY,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    manifest = load_manifest(MANIFEST_PATH)

    # Remove stale source vectors if a previously indexed source is no longer configured.
    current_source_keys = {source_key(s) for s in sources}
    stale_keys = [key for key in list(manifest.keys()) if key not in current_source_keys]
    for key in stale_keys:
        vectorstore.delete(where={"source_key": key})
        del manifest[key]
        log_event("source_removed", source_key=key)

    changed_count = 0
    skipped_count = 0
    total_chunks_indexed = 0

    for source in sources:
        key = source_key(source)
        fingerprint = source_fingerprint(source)

        # Fast path: unchanged source can be reused from existing index.
        if manifest.get(key) == fingerprint:
            skipped_count += 1
            log_event("source_skipped", source_key=key)
            continue

        docs = read_pdf_pages(
            pdf_path=source["pdf_path"],
            start_page=source["start_page"],
            end_page=source["end_page"],
        )
        if not docs:
            log_event("source_empty", source_key=key)
            continue

        # Attach source_key to every page/chunk so bulk-delete by source is easy later.
        for doc in docs:
            doc.metadata["source_key"] = key

        chunk_docs = splitter.split_documents(docs)

        # Replace only this source's vectors (idempotent update behavior).
        vectorstore.delete(where={"source_key": key})
        vectorstore.add_documents(chunk_docs)

        manifest[key] = fingerprint
        changed_count += 1
        total_chunks_indexed += len(chunk_docs)

        log_event(
            "source_reindexed",
            source_key=key,
            page_count=len(docs),
            chunk_count=len(chunk_docs),
        )

    save_manifest(MANIFEST_PATH, manifest)
    log_event(
        "indexing_summary",
        changed_sources=changed_count,
        skipped_sources=skipped_count,
        total_chunks_indexed=total_chunks_indexed,
    )

    return vectorstore

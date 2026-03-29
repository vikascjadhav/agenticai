# RAG With Store (Modular)

A modular Retrieval-Augmented Generation (RAG) application that:
- reads selected PDF page ranges,
- indexes chunks in Chroma,
- supports incremental re-indexing with a manifest,
- retrieves relevant chunks for user questions,
- generates grounded answers through a local OpenAI-compatible model endpoint,
- emits structured JSON logs.

## Project Structure

```text
rag-with-store.py             # thin entrypoint
rag_app/
  __init__.py                 # package overview
  cli.py                      # interactive app orchestration
  config.py                   # constants + source/query presets
  clients.py                  # embedding + chat model clients
  pdf_loader.py               # PDF page extraction to LangChain Document
  manifest.py                 # fingerprinting + manifest persistence
  indexer.py                  # incremental Chroma indexing
  rag.py                      # prompt chain + context formatting + citations
chroma_db/                    # persistent Chroma store + manifest
```

## Runtime Flow

1. `rag-with-store.py` calls `rag_app.cli.run_app()`.
2. `run_app()` configures structured logging.
3. Sources are loaded from `config.get_sources()`.
4. `indexer.load_vector_store()`:
   - opens Chroma,
   - loads manifest,
   - removes stale source vectors,
   - re-indexes only changed sources,
   - saves manifest.
5. Retriever + RAG chain are built.
6. CLI loop accepts preset or custom query.
7. App retrieves top-k chunks, formats context, runs LLM, logs result metadata.

## Incremental Indexing (Manifest)

Manifest path: `./chroma_db/index_manifest.json`

### Fingerprint Inputs
Each source fingerprint hashes:
- absolute `pdf_path`
- file `mtime_ns`
- file `size`
- `start_page`
- `end_page`

### Behavior
- Unchanged fingerprint => source skipped
- Changed fingerprint => source re-indexed
- Source removed from config => vectors deleted + manifest key removed

This makes startup fast for stable documents.

## Structured Logging

Logs are emitted as JSON lines to stdout.

### Key Events
- `app_start`, `app_end`
- `source_reindexed`
- `source_skipped`
- `source_removed`
- `indexing_summary`
- `query_processing_started`
- `query_completed`
- `invalid_input`

### Example
```json
{"event":"query_completed","query":"what is the repo rate?","latency_ms":845.13,"retrieved_count":3,"citations":["RBI-Monetory-Policy-OCT-2025.pdf (page 10)"]}
```

## Prerequisites

- Windows PowerShell (or any shell with Python)
- Python virtual environment
- Local OpenAI-compatible endpoint available at:
  - `http://localhost:1234/v1`
- PDF file available in project root:
  - `RBI-Monetory-Policy-OCT-2025.pdf`

## Install

From project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If needed, ensure `pypdf` is installed:

```powershell
.\.venv\Scripts\python.exe -m pip install pypdf
```

## Run

Option 1 (entrypoint):

```powershell
.\.venv\Scripts\python.exe rag-with-store.py
```

Option 2 (module):

```powershell
.\.venv\Scripts\python.exe -m rag_app.cli
```

## Configuration

Edit `rag_app/config.py`:

- `get_sources()` for PDFs/page ranges
- `COLLECTION_NAME`
- `CHUNK_SIZE`, `CHUNK_OVERLAP`
- `RETRIEVAL_K`
- `get_queries()` for menu items

Embedding model can be overridden with env var:

```powershell
$env:LOCAL_EMBED_MODEL="text-embedding-nomic-embed-text-v1.5"
```

## Notes on Query Menu

In `get_queries()`:
- second-last item is treated as custom question option,
- last item is treated as exit option.

If you change list order, update this convention in `rag_app/cli.py`.

## Troubleshooting

### 1) App hangs on query
- Ensure local model server at `http://localhost:1234/v1` is running.
- Verify selected chat/embedding models are loaded and available.

### 2) No results / weak answers
- Increase `RETRIEVAL_K`
- Tune chunking (`CHUNK_SIZE`, `CHUNK_OVERLAP`)
- Expand source page ranges in `get_sources()`

### 3) Re-index not triggering
- Confirm file actually changed (mtime/size), or page range changed.
- Delete `chroma_db/index_manifest.json` to force fresh indexing.

### 4) Python warning about Pydantic v1 on 3.14
- Current dependencies may warn on Python 3.14.
- Python 3.12/3.13 is typically safer for LangChain compatibility.

## Extension Ideas

- Add API layer (FastAPI) on top of `run_app` services.
- Add timeout/retry wrappers for embedding/LLM requests.
- Add answer confidence and retrieval score logging.
- Add unit/integration tests for manifest/index behavior.


## OLD Code for reference if needed

Option 1 (entrypoint):

```python
from langchain_chroma import Chroma
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from pypdf import PdfReader

MANIFEST_PATH = Path("./chroma_db/index_manifest.json")
COLLECTION_NAME = "rbi_policy_oct_2025"
LOGGER = logging.getLogger("rag_with_store")


def configure_logging() -> None:
    """Emit structured JSON logs to stdout."""
    if LOGGER.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def log_event(event: str, level: str = "info", **fields: Any) -> None:
    payload = {"event": event, **fields}
    message = json.dumps(payload, ensure_ascii=False, default=str)
    getattr(LOGGER, level, LOGGER.info)(message)


def get_sources() -> list[dict[str, Any]]:
    """Configure PDF sources and page ranges to index."""
    return [
        {
            "pdf_path": "RBI-Monetory-Policy-OCT-2025.pdf",
            "start_page": 9,
            "end_page": 19,
        },
    ]


def source_key(source: dict[str, Any]) -> str:
    """Stable source identifier used in manifest and vector metadata."""
    abs_path = str(Path(source["pdf_path"]).resolve())
    return f"{abs_path}|{source['start_page']}-{source['end_page']}"


def source_fingerprint(source: dict[str, Any]) -> str:
    """
    Fingerprint changes that should trigger re-indexing.
    We hash path + mtime + size + page range.
    """
    abs_path = str(Path(source["pdf_path"]).resolve())
    stat = os.stat(abs_path)
    payload = {
        "path": abs_path,
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
        "start_page": source["start_page"],
        "end_page": source["end_page"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def load_manifest() -> dict[str, str]:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def save_manifest(manifest: dict[str, str]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_pdf_documents(sources: list[dict[str, Any]]) -> list[Document]:
    """
    Load pages from all configured PDFs into one flat list of LangChain Documents.

    Returns:
        list[Document]:
            One Document per extracted PDF page.
            - Document.page_content -> page text
            - Document.metadata["id"] -> stable page id
            - Document.metadata["page"] -> 1-based page number
            - Document.metadata["source"] -> PDF file name
    """
    all_docs: list[Document] = []
    for source in sources:
        all_docs.extend(
            read_pdf_pages(
                pdf_path=source["pdf_path"],
                start_page=source["start_page"],
                end_page=source["end_page"],
            )
        )

    log_event(
        "pdf_pages_loaded",
        source_count=len(sources),
        page_count=len(all_docs),
    )
    return all_docs


def read_pdf_pages(
    pdf_path: str,
    start_page: int,
    end_page: int,
) -> list[Document]:
    """
    Read one PDF and return one Document per selected page.

    This function is intentionally scoped to a single PDF.
    `load_pdf_documents` is responsible for aggregating many PDFs.
    """
    if not pdf_path:
        raise ValueError("pdf_path must be a non-empty string")
    if start_page < 1:
        raise ValueError("start_page must be >= 1")
    if end_page < start_page:
        raise ValueError("end_page must be >= start_page")

    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    if start_page > total_pages:
        raise ValueError(
            f"start_page {start_page} exceeds total pages {total_pages} in {pdf_path}"
        )

    safe_end_page = min(end_page, total_pages)
    file_name = os.path.basename(pdf_path)
    docs: list[Document] = []

    # Convert user-facing 1-based page numbers into pypdf's 0-based index.
    for page_num in range(start_page, safe_end_page + 1):
        page_text = (reader.pages[page_num - 1].extract_text() or "").strip()
        if not page_text:
            continue

        docs.append(
            Document(
                page_content=page_text,
                metadata={
                    "id": f"{file_name}:page-{page_num}",
                    "page": page_num,
                    "source": file_name,
                },
            )
        )

    return docs



def local_embeddings() -> OpenAIEmbeddings:
    """
    Create embeddings client for a local OpenAI-compatible server.

    LOCAL_EMBED_MODEL lets you switch the embedding model at runtime:
        $env:LOCAL_EMBED_MODEL="text-embedding-nomic-embed-text-v1.5"
    """
    return OpenAIEmbeddings(
        base_url="http://localhost:1234/v1",
        model=os.getenv("LOCAL_EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5"),
        api_key="fake-api-key",
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


def load_vector_store() -> Optional[Chroma]:
    sources = get_sources()
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=local_embeddings(),
        persist_directory="./chroma_db",
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    manifest = load_manifest()

    current_source_keys = {source_key(s) for s in sources}
    stale_keys = [key for key in manifest.keys() if key not in current_source_keys]
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
            continue

        # Tag each page with source_key so old chunks can be removed by source.
        for doc in docs:
            doc.metadata["source_key"] = key

        chunk_docs = splitter.split_documents(docs)
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

    save_manifest(manifest)
    log_event(
        "indexing_summary",
        changed_sources=changed_count,
        skipped_sources=skipped_count,
        total_chunks_indexed=total_chunks_indexed,
    )

    return vectorstore

def format_retrieved_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"{doc.metadata['source']} (page {doc.metadata['page']}): {doc.page_content}" for doc in docs)

def main() -> None:
    # Avoid Windows console crashes when model output includes unsupported Unicode chars.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    configure_logging()

    log_event("app_start")

    vectorstore = load_vector_store()  # one time setup to build the vector store
    if vectorstore is None:
        log_event("vectorstore_init_failed", level="error")
        return


    prompt_template = ChatPromptTemplate.from_template(
    "Use only the context below.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    'If the answer is not present, say "I don\'t know."')
    
    queries = [
        "can you summarize RBI policy?",
        "what is the repo rate?",
        "what is the GDP growth forecast?",
        "what is the inflation forecast?",
        "what is agentic AI?",
        "Other Text",
        "exit"
    ]
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = local_llm()
    rag_chain =   prompt_template | llm | StrOutputParser()

    repeat = True
    while repeat:
        options = "\n".join(f"{i}. {query}" for i, query in enumerate(queries, 1))
        user_input = input(
            "\nEnter a question number (or 'exit'):\n"
            f"{options}\n> "
        )
        if user_input.lower() == 'exit' or user_input == str(len(queries)):
            repeat = False
            break
        elif user_input.isdigit() and (len(queries)-1) == int(user_input):
            user_input = input("\n\nEnter your custom query: ")
        elif user_input.isdigit() and 1 <= int(user_input) <= len(queries):
            user_input = queries[int(user_input) - 1]
        else:
            log_event("invalid_input", input=user_input)
            continue
        
        started = time.perf_counter()
        log_event("query_processing_started", query=user_input)
        retrieved_docs = retriever.invoke(user_input) 
        context = format_retrieved_docs(retrieved_docs)
        answer = rag_chain.invoke({"question": user_input, "context": context})
        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        citations = sorted(
            {
                f"{doc.metadata.get('source', 'unknown')} (page {doc.metadata.get('page', '?')})"
                for doc in retrieved_docs
            }
        )
        log_event(
            "query_completed",
            query=user_input,
            latency_ms=latency_ms,
            retrieved_count=len(retrieved_docs),
            citations=citations,
            answer_preview=str(answer)[:300],
        )
        
    log_event("app_end")

if __name__ == "__main__":
    main()

```

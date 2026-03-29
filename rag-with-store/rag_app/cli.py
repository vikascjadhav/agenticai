"""CLI orchestration for the RAG application.

This module wires all services together:
- logging setup
- vector store loading/index refresh
- retriever + RAG chain execution
- interactive menu loop
"""

import sys
import time

from rag_app.clients import local_llm
from rag_app.config import RETRIEVAL_K, get_queries, get_sources
from rag_app.indexer import load_vector_store
from rag_app.logging_utils import configure_logging, log_event
from rag_app.rag import build_rag_chain, citations_from_docs, format_retrieved_docs


def run_app() -> None:
    """Run interactive RAG CLI loop until user exits."""
    # Prevent Unicode print failures on Windows terminals with limited encodings.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    configure_logging()
    log_event("app_start")

    # 1) Build/update index once at startup.
    sources = get_sources()
    vectorstore = load_vector_store(sources)

    # 2) Build retrieval and generation pipeline.
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    rag_chain = build_rag_chain(local_llm())

    # 3) Start interactive question loop.
    queries = get_queries()

    while True:
        options = "\n".join(f"{i}. {query}" for i, query in enumerate(queries, 1))
        user_input = input(
            "\nEnter a question number (or 'exit'):\n"
            f"{options}\n> "
        ).strip()

        # Exit conditions: typed 'exit' OR selected last menu item.
        if user_input.lower() == "exit" or user_input == str(len(queries)):
            break

        # Convention: second-last menu entry is custom free-text input.
        if user_input.isdigit() and int(user_input) == len(queries) - 1:
            user_input = input("\nEnter your custom query:\n> ").strip()

        # Map numeric menu choice to preset question.
        elif user_input.isdigit() and 1 <= int(user_input) <= len(queries):
            user_input = queries[int(user_input) - 1]

        # Reject empty/invalid input and continue loop.
        elif not user_input:
            log_event("invalid_input", input=user_input)
            continue

        started = time.perf_counter()
        log_event("query_processing_started", query=user_input)

        # Retrieve top-k supporting chunks and format context text.
        retrieved_docs = retriever.invoke(user_input)
        context = format_retrieved_docs(retrieved_docs)

        # Generate final answer from context + question.
        answer = rag_chain.invoke({"question": user_input, "context": context})

        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        log_event(
            "query_completed",
            query=user_input,
            latency_ms=latency_ms,
            retrieved_count=len(retrieved_docs),
            citations=citations_from_docs(retrieved_docs),
            answer_preview=str(answer)[:300],
        )

    log_event("app_end")

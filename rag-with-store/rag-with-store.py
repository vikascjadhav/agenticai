"""Project entrypoint.

This file intentionally stays minimal so the application can be launched with:
    python rag-with-store.py

All runtime logic lives in `rag_app.cli.run_app`.
"""

from rag_app.cli import run_app


if __name__ == "__main__":
    # Delegate full startup/index/query loop to the CLI orchestrator.
    run_app()

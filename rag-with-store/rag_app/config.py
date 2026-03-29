"""Central configuration for the RAG application.

Why this module exists:
- Keeps tuning knobs in one place.
- Avoids hard-coded values scattered across indexing/query code.
- Makes future environment-based config migration straightforward.
"""

from pathlib import Path
from typing import Any

# Path to manifest that tracks source fingerprints (used for incremental indexing).
MANIFEST_PATH = Path("./chroma_db/index_manifest.json")

# Chroma persistence directory on local disk.
PERSIST_DIRECTORY = "./chroma_db"

# Chroma collection name for this app.
COLLECTION_NAME = "rbi_policy_oct_2025"

# Text splitter settings used during indexing.
# Increase CHUNK_SIZE for more context per chunk; increase overlap for continuity.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Number of retrieved chunks per query.
RETRIEVAL_K = 3


def get_sources() -> list[dict[str, Any]]:
    """Return configured PDF sources and page ranges to ingest.

    Output contract (each dict):
    - pdf_path: local file path to PDF
    - start_page: 1-based inclusive page start
    - end_page: 1-based inclusive page end

    Add more entries to scale ingestion to multiple PDFs.
    """
    return [
        {
            "pdf_path": "RBI-Monetory-Policy-OCT-2025.pdf",
            "start_page": 9,
            "end_page": 19,
        },
    ]


def get_queries() -> list[str]:
    """Return preset CLI menu questions.

    Convention:
    - Second-last item is "Other Text" (custom query path).
    - Last item is "exit" (menu exit shortcut).
    """
    return [
        "can you summarize RBI policy?",
        "what is the repo rate?",
        "what is the GDP growth forecast?",
        "what is the inflation forecast?",
        "what is agentic AI?",
        "Other Text",
        "exit",
    ]

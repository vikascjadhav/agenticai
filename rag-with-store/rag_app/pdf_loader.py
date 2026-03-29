"""PDF loading utilities.

Converts selected PDF pages into LangChain `Document` objects with metadata.
"""

import os

from langchain_core.documents import Document
from pypdf import PdfReader


def read_pdf_pages(pdf_path: str, start_page: int, end_page: int) -> list[Document]:
    """Read one PDF and return one `Document` per page in the requested range.

    Args:
    - pdf_path: local PDF file path
    - start_page: 1-based inclusive start page
    - end_page: 1-based inclusive end page

    Returns:
    - list[Document], where each document contains:
      - page_content: extracted page text
      - metadata.id: stable id `filename:page-N`
      - metadata.page: 1-based page number
      - metadata.source: PDF filename

    Raises:
    - ValueError for invalid input range or missing path
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

    # Convert user-facing 1-based page numbers to pypdf's 0-based indexes.
    for page_num in range(start_page, safe_end_page + 1):
        page_text = (reader.pages[page_num - 1].extract_text() or "").strip()

        # Skip pages that extract as empty to avoid indexing noise.
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

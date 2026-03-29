"""Structured logging helpers.

This module emits JSON logs so logs can be parsed by tools and log pipelines
without relying on fragile string parsing.
"""

import json
import logging
import sys
from typing import Any

# Shared logger for the whole application.
LOGGER = logging.getLogger("rag_with_store")


def configure_logging() -> None:
    """Initialize one stdout handler that emits raw JSON lines.

    Idempotent behavior:
    - If handlers already exist, we do nothing.
    - This prevents duplicate log lines during repeated setup calls.
    """
    if LOGGER.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)

    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def log_event(event: str, level: str = "info", **fields: Any) -> None:
    """Log a structured event.

    Args:
    - event: short event name (for example, "query_completed")
    - level: logging level name supported by logger (default: "info")
    - fields: arbitrary key/value payload for the event
    """
    payload = {"event": event, **fields}
    message = json.dumps(payload, ensure_ascii=False, default=str)
    getattr(LOGGER, level, LOGGER.info)(message)

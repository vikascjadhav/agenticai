"""Manifest and fingerprint helpers for incremental indexing.

The manifest maps a stable source key to a fingerprint hash. If the fingerprint
changes, we re-index that source. If unchanged, we skip it.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def source_key(source: dict[str, Any]) -> str:
    """Build stable key from absolute file path + page range.

    Key format:
    <abs_path>|<start_page>-<end_page>
    """
    abs_path = str(Path(source["pdf_path"]).resolve())
    return f"{abs_path}|{source['start_page']}-{source['end_page']}"


def source_fingerprint(source: dict[str, Any]) -> str:
    """Return SHA-256 fingerprint for change detection.

    Fingerprint input includes:
    - absolute file path
    - file modification time (ns)
    - file size
    - start/end page range

    Any change to these fields triggers re-indexing.
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
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_manifest(manifest_path: Path) -> dict[str, str]:
    """Load manifest JSON from disk. Returns empty map if file is absent."""
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def save_manifest(manifest_path: Path, manifest: dict[str, str]) -> None:
    """Persist manifest map to disk (creates parent directory if needed)."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

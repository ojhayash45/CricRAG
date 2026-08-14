#!/usr/bin/env python
"""
Rebuild only the embeddings + FAISS index from already-ingested chunks.

    python scripts/build_index.py

Use this after changing EMBEDDING_MODEL, or similar -- it's the fast path
that skips re-loading/re-cleaning/re-parsing the source corpus. Run
scripts/ingest.py first if data/chunks/chunks.json doesn't exist yet.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.ingest_runner import rebuild_index_from_saved_chunks  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_index")


def main() -> int:
    try:
        embeddings_generated = rebuild_index_from_saved_chunks()
    except RuntimeError as exc:
        logger.error(str(exc))
        return 1

    print()
    print(f"Embeddings generated: {embeddings_generated}")
    print("Vector index built successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

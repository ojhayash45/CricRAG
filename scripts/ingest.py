#!/usr/bin/env python
"""
Full ingestion CLI: source -> clean -> parse -> chunk -> embed -> FAISS index.

    python scripts/ingest.py

Reads whatever authorized .txt/.html/.pdf files are in data/raw/ -- see
README's copyright section for why this is local-file-only, with no live
scraping path.

Chunks are also persisted to data/chunks/chunks.json so that:
  - scripts/inspect_chunks.py can inspect them without re-running ingestion
  - scripts/build_index.py (or the Streamlit sidebar's "Rebuild index") can
    rebuild just the embeddings/index without re-parsing the source files

The actual pipeline logic lives in src/pipeline/ingest_runner.py, shared
with app.py, so there is exactly one implementation of it.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.ingest_runner import run_ingestion  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ingest")


def main() -> int:
    try:
        stats = run_ingestion()
    except RuntimeError as exc:
        logger.error(str(exc))
        return 1

    print()
    print(f"Documents loaded: {stats.documents_loaded}")
    print(f"Laws found: {stats.laws_found}")
    print(f"Sections found: {stats.sections_found}")
    print(f"Chunks created: {stats.chunks_created}")
    print(f"Embeddings generated: {stats.embeddings_generated}")
    print("Vector index built successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

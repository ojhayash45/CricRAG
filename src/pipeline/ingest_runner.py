"""
Shared ingestion pipeline logic.

Both scripts/ingest.py (CLI) and app.py's sidebar "Rebuild index" control
call into this module, so there is exactly one implementation of
load -> dedupe -> chunk -> save -> embed -> index -> save index. Streamlit
must never re-embed on every rerun (spec section 36) -- callers are
responsible for only invoking these functions on an explicit user action,
not on every script/page load.

Ingestion is local-corpus-only: place authorized .txt/.html/.pdf files in
data/raw/ and run this. There is no live-scraping path -- see README's
copyright/source section for why.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from config import settings
from src.chunking.chunker import SectionAwareChunker
from src.embeddings.embedding_service import EmbeddingService
from src.ingestion.cleaner import deduplicate_by_hash
from src.ingestion.document_loader import LocalDocumentLoader
from src.ingestion.parser import parse_law_document
from src.models.schemas import Chunk, Document
from src.vectorstore.faiss_store import FAISSStore

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    documents_loaded: int
    laws_found: int
    sections_found: int
    chunks_created: int
    embeddings_generated: int


def load_documents() -> list[Document]:
    """Step 1-3: validate source configuration and load raw documents from data/raw/."""
    logger.info("Using local corpus loader (data/raw/): %s", settings.raw_data_dir)
    return LocalDocumentLoader().load()


def save_chunks(chunks: list[Chunk]) -> None:
    settings.chunks_data_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.chunks_data_dir / "chunks.json"
    out_path.write_text(
        json.dumps([c.model_dump(mode="json") for c in chunks], indent=2),
        encoding="utf-8",
    )
    logger.info("Saved %d chunk(s) to %s", len(chunks), out_path)


def load_saved_chunks() -> list[Chunk]:
    chunks_path = settings.chunks_data_dir / "chunks.json"
    if not chunks_path.exists():
        raise RuntimeError(f"No chunks found at {chunks_path}. Run ingestion first.")
    payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    return [Chunk.model_validate(item) for item in payload]


def run_ingestion() -> IngestionStats:
    """Full pipeline: load -> dedupe -> chunk -> save chunks -> embed -> build+save FAISS index."""
    documents = load_documents()
    if not documents:
        raise RuntimeError(
            f"No documents were loaded. Place an authorized corpus (.txt/.html/.pdf) in "
            f"{settings.raw_data_dir} (see README's copyright/source section), then try again."
        )

    documents = deduplicate_by_hash(documents, text_getter=lambda d: d.content)
    logger.info("Documents loaded: %d", len(documents))

    laws_found = {d.law_number for d in documents if d.law_number}
    sections_found = 0
    for doc in documents:
        parsed = parse_law_document(doc.content)
        sections_found += sum(1 for s in parsed.sections if s.section)

    chunker = SectionAwareChunker()
    chunks = chunker.chunk_documents(documents)
    logger.info("Chunks created: %d", len(chunks))
    save_chunks(chunks)

    if not chunks:
        raise RuntimeError("No chunks were produced -- nothing to index.")

    embeddings_generated = _embed_and_index(chunks)

    return IngestionStats(
        documents_loaded=len(documents),
        laws_found=len(laws_found),
        sections_found=sections_found,
        chunks_created=len(chunks),
        embeddings_generated=embeddings_generated,
    )


def rebuild_index_from_saved_chunks() -> int:
    """Fast path: re-embed + re-index chunks already saved by a previous run_ingestion() call."""
    chunks = load_saved_chunks()
    if not chunks:
        raise RuntimeError("Saved chunks file is empty.")
    return _embed_and_index(chunks)


def _embed_and_index(chunks: list[Chunk]) -> int:
    logger.info("Generating embeddings for %d chunk(s) using %s ...", len(chunks), settings.embedding_model)
    embedder = EmbeddingService()
    embeddings = embedder.embed_documents([c.text for c in chunks])
    logger.info("Embeddings generated: %d", embeddings.shape[0])

    store = FAISSStore(dimension=embedder.dimension)
    store.add(embeddings, chunks)
    store.save()
    logger.info("Vector index built successfully.")
    return int(embeddings.shape[0])

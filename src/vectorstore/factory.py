"""
Picks the vector store backend based on config, so every caller (ingestion,
the Streamlit app, evaluation) uses the same rule instead of each deciding
for itself: Postgres if DATABASE_URL is configured, local FAISS files
otherwise. See postgres_store.py / faiss_store.py for why each exists.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from config import settings
from src.models.schemas import Chunk
from src.vectorstore.faiss_store import SearchResult


class VectorStore(Protocol):
    """Structural interface both FAISSStore and PostgresVectorStore satisfy."""

    def add(self, embeddings: np.ndarray, chunks: list[Chunk]) -> int: ...
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[SearchResult]: ...
    def save(self) -> None: ...
    @property
    def size(self) -> int: ...
    @property
    def chunks(self) -> list[Chunk]: ...


def build_fresh_vector_store(dimension: int) -> VectorStore:
    """Used during (re)ingestion -- always starts from a clean slate."""
    if settings.database_url:
        from src.vectorstore.postgres_store import PostgresVectorStore
        store = PostgresVectorStore()
        store.clear()
        return store

    from src.vectorstore.faiss_store import FAISSStore
    return FAISSStore(dimension=dimension)


def load_vector_store() -> VectorStore | None:
    """
    Used at app boot / query time -- load whatever already exists.

    Returns None only for the FAISS case with no index file yet; the
    Postgres case returns a real (possibly empty, size == 0) store, since
    "connected but empty" and "not configured" are different situations.
    Every caller already checks `size > 0` before treating a store as
    ready, so both cases behave the same from there.
    """
    if settings.database_url:
        from src.vectorstore.postgres_store import PostgresVectorStore
        return PostgresVectorStore()

    from src.vectorstore.faiss_store import FAISSStore
    try:
        return FAISSStore.load()
    except FileNotFoundError:
        return None

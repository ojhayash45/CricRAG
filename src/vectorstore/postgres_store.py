"""
Postgres + pgvector backed vector store.

Same job as FAISSStore (pair a vector with the Chunk it came from, support
add/search), but persistent in a real database instead of local files --
this is what lets a deployed instance survive a restart/redeploy without
losing its index, and what lets the full real corpus (all 42 laws, 886
chunks) be populated once from a local machine and then just be *there*
for every deployment, with no PDFs and no re-ingestion needed on the
server at all.

Chosen over local files specifically because Render's free tier has no
persistent disk -- everything written to disk is gone on the next restart.

No connection pool: each method opens and closes its own short-lived
connection. At this scale (hundreds of chunks, occasional queries from one
Streamlit process) that's simpler and plenty fast; a pool would be the
right next step if this were serving concurrent traffic at real volume.

No ANN index (ivfflat/hnsw) on the embedding column either -- an exact
sequential scan over a few hundred/thousand rows is already sub-10ms.
That's a genuine future scaling concern (see README), not an oversight.
"""
from __future__ import annotations

import logging

import numpy as np

from config import settings
from src.models.schemas import Chunk
from src.vectorstore.faiss_store import SearchResult

logger = logging.getLogger(__name__)

DEFAULT_TABLE = "chunks"

_COLUMNS = (
    "chunk_id", "document_id", "law_number", "law_title", "section",
    "edition", "text", "source_url", "chunk_index", "content_hash",
)


class PostgresVectorStore:
    def __init__(self, database_url: str | None = None, table: str = DEFAULT_TABLE):
        """
        table is configurable (not just hard-coded to "chunks") so tests can
        point at an isolated, disposable table name instead of ever risking
        the real indexed corpus -- even if a test run happens to point at
        the same database as production.
        """
        self.database_url = database_url or settings.database_url
        if not self.database_url:
            raise ValueError(
                "PostgresVectorStore requires a connection string -- set DATABASE_URL in .env."
            )
        self.table = table

    def _connect(self):
        import psycopg
        from pgvector.psycopg import register_vector

        conn = psycopg.connect(self.database_url, autocommit=True)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        return conn

    @property
    def size(self) -> int:
        import psycopg

        try:
            with self._connect() as conn:
                row = conn.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()
                return int(row[0]) if row else 0
        except psycopg.errors.UndefinedTable:
            return 0

    @property
    def chunks(self) -> list[Chunk]:
        """All indexed chunks, for UI/stats/evaluation use -- not the hot query path."""
        import psycopg

        cols = ", ".join(_COLUMNS)
        try:
            with self._connect() as conn:
                rows = conn.execute(f"SELECT {cols} FROM {self.table} ORDER BY document_id, chunk_index").fetchall()
        except psycopg.errors.UndefinedTable:
            return []
        return [_row_to_chunk(row) for row in rows]

    def clear(self) -> None:
        """Drop all indexed data -- used before a full rebuild so it doesn't accumulate stale rows."""
        with self._connect() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {self.table}")
        logger.info("Cleared Postgres vector store table %r", self.table)

    def save(self) -> None:
        """No-op: every add() is already a committed INSERT. Kept for interface parity with FAISSStore."""

    def add(self, embeddings: np.ndarray, chunks: list[Chunk]) -> int:
        """
        Add embeddings + their source chunks. Chunks whose content_hash was
        already indexed are skipped (corpus-level duplicate protection, via
        a unique constraint + ON CONFLICT DO NOTHING). Returns the number
        of rows actually inserted.
        """
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D (n, dimension), got shape {embeddings.shape}")
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"embeddings row count ({embeddings.shape[0]}) must match chunk count ({len(chunks)})"
            )

        dimension = embeddings.shape[1]
        placeholders = ", ".join(["%s"] * len(_COLUMNS))

        with self._connect() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    law_number TEXT,
                    law_title TEXT,
                    section TEXT,
                    edition TEXT,
                    text TEXT NOT NULL,
                    source_url TEXT,
                    chunk_index INTEGER NOT NULL DEFAULT 0,
                    content_hash TEXT UNIQUE,
                    embedding vector({dimension})
                )
            """)

            added = 0
            for vec, chunk in zip(embeddings, chunks):
                values = tuple(getattr(chunk, col) for col in _COLUMNS)
                cur = conn.execute(
                    f"""
                    INSERT INTO {self.table} ({", ".join(_COLUMNS)}, embedding)
                    VALUES ({placeholders}, %s)
                    ON CONFLICT (content_hash) DO NOTHING
                    """,
                    (*values, np.asarray(vec, dtype=np.float32)),
                )
                added += cur.rowcount

        logger.info("Added %d row(s) to Postgres vector store (skipped %d duplicate(s))",
                    added, len(chunks) - added)
        return added

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        """Return up to top_k (chunk, score) results, highest cosine similarity first."""
        import psycopg

        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        cols = ", ".join(_COLUMNS)

        try:
            with self._connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT {cols}, 1 - (embedding <=> %s) AS score
                    FROM {self.table}
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (query, query, top_k),
                ).fetchall()
        except psycopg.errors.UndefinedTable:
            return []

        return [SearchResult(chunk=_row_to_chunk(row[:-1]), score=float(row[-1])) for row in rows]


def _row_to_chunk(row) -> Chunk:
    return Chunk(**dict(zip(_COLUMNS, row)))

"""
FAISS-backed vector store.

Every vector added is paired 1:1 with the Chunk it came from -- the FAISS
index only ever holds numbers, so this class is what makes a similarity
result mean something (which law, which section, which source URL).

On disk this is two files that must always be read/written together:
    vector_store/index.faiss    -- the raw FAISS index
    vector_store/metadata.json  -- {dimension, metadata[]} in the same
                                    row order as vectors in the index

IndexFlatIP (inner product) is used deliberately: EmbeddingService
L2-normalizes every vector, so inner product is equivalent to cosine
similarity, and scores are directly comparable to SIMILARITY_THRESHOLD.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from config import settings
from src.models.schemas import Chunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class FAISSStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._metadata: list[dict] = []
        self._seen_hashes: set[str] = set()

    @property
    def size(self) -> int:
        return self._index.ntotal

    @property
    def chunks(self) -> list[Chunk]:
        """All indexed chunks, for UI/stats/evaluation use -- not the hot query path."""
        return [Chunk.model_validate(m) for m in self._metadata]

    def add(self, embeddings: np.ndarray, chunks: list[Chunk]) -> int:
        """
        Add embeddings + their source chunks. Chunks whose content_hash was
        already indexed are skipped (corpus-level duplicate protection).
        Returns the number of vectors actually added.
        """
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D (n, dimension), got shape {embeddings.shape}")
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"embeddings row count ({embeddings.shape[0]}) must match chunk count ({len(chunks)})"
            )
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"embedding dimension ({embeddings.shape[1]}) does not match store dimension ({self.dimension})"
            )

        keep_indices: list[int] = []
        for i, chunk in enumerate(chunks):
            if chunk.content_hash and chunk.content_hash in self._seen_hashes:
                logger.info("Skipping duplicate chunk %s (hash already indexed)", chunk.chunk_id)
                continue
            keep_indices.append(i)
            if chunk.content_hash:
                self._seen_hashes.add(chunk.content_hash)

        if not keep_indices:
            return 0

        filtered_embeddings = np.ascontiguousarray(embeddings[keep_indices], dtype=np.float32)
        self._index.add(filtered_embeddings)
        self._metadata.extend(chunks[i].model_dump(mode="json") for i in keep_indices)

        logger.info("Added %d vector(s) to index (skipped %d duplicate(s))",
                    len(keep_indices), len(chunks) - len(keep_indices))
        return len(keep_indices)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        """Return up to top_k (chunk, score) results, highest similarity first."""
        if self._index.ntotal == 0:
            return []

        query = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)
        if query.shape[1] != self.dimension:
            raise ValueError(
                f"query embedding dimension ({query.shape[1]}) does not match store dimension ({self.dimension})"
            )

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query, k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = Chunk.model_validate(self._metadata[idx])
            results.append(SearchResult(chunk=chunk, score=float(score)))
        return results

    def save(self, index_path: Path | None = None, metadata_path: Path | None = None) -> None:
        index_path = index_path or (settings.vector_store_dir / "index.faiss")
        metadata_path = metadata_path or (settings.vector_store_dir / "metadata.json")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(index_path))
        metadata_path.write_text(
            json.dumps({"dimension": self.dimension, "metadata": self._metadata}, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved FAISS index (%d vectors) to %s and metadata to %s",
                    self.size, index_path, metadata_path)

    @classmethod
    def load(cls, index_path: Path | None = None, metadata_path: Path | None = None) -> "FAISSStore":
        index_path = index_path or (settings.vector_store_dir / "index.faiss")
        metadata_path = metadata_path or (settings.vector_store_dir / "metadata.json")

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Vector store not found at {index_path} / {metadata_path}. "
                "Run the ingestion/index-build script first."
            )

        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        dimension = payload["dimension"]
        metadata = payload["metadata"]

        store = cls(dimension=dimension)
        store._index = faiss.read_index(str(index_path))
        store._metadata = metadata
        store._seen_hashes = {m["content_hash"] for m in metadata if m.get("content_hash")}

        if store._index.ntotal != len(store._metadata):
            raise ValueError(
                f"Loaded index/metadata are out of sync: {store._index.ntotal} vectors "
                f"vs {len(store._metadata)} metadata entries"
            )

        logger.info("Loaded FAISS index with %d vectors from %s", store.size, index_path)
        return store

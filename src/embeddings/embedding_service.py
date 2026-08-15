"""
Embedding layer.

A thin, explicit wrapper around a text embedding model. Its only job is
text -> vector, done consistently for both indexing and querying, so the
rest of the pipeline (FAISS store, retriever) never touches a model
directly. The model name is fully configurable via
config.settings.embedding_model -- swapping models means changing config,
not code.

Uses fastembed (ONNX Runtime) rather than sentence-transformers (PyTorch).
Same default model (sentence-transformers/all-MiniLM-L6-v2, same 384-dim
output), but ONNX Runtime has a much smaller memory footprint than torch --
this matters on memory-constrained hosts (e.g. a 512MB free-tier deploy),
where importing/loading torch alone was enough to exhaust available RAM.
"""
from __future__ import annotations

import logging

import numpy as np

from config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str | None = None, normalize: bool = True):
        """
        fastembed's models are already L2-normalized on output, so
        normalize=True here is mostly documentation of that fact -- kept
        as a parameter for interface compatibility and so a future model
        that *isn't* pre-normalized can still be supported correctly.

        The fastembed import is deferred to __init__ (not module level)
        for the same reason the old torch-based import was: keep it out
        of the process's baseline memory until an EmbeddingService is
        actually constructed.
        """
        from fastembed import TextEmbedding

        self.model_name = model_name or settings.embedding_model
        self.normalize = normalize
        logger.info("Loading embedding model: %s", self.model_name)
        self._model = TextEmbedding(model_name=self.model_name)
        self._dimension = self._detect_dimension()

    def _detect_dimension(self) -> int:
        from fastembed import TextEmbedding

        for entry in TextEmbedding.list_supported_models():
            if entry["model"] == self.model_name:
                return int(entry["dim"])
        # Fallback for a model not in fastembed's known-models table.
        probe = next(iter(self._model.embed(["dimension probe"])))
        return int(probe.shape[-1])

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Batch-embed chunk/document texts. Returns shape (len(texts), dimension)."""
        if not texts:
            raise ValueError("embed_documents() requires at least one text")
        cleaned = [t.strip() for t in texts]
        if any(not t for t in cleaned):
            raise ValueError("embed_documents() received an empty/whitespace-only text")

        embeddings = list(self._model.embed(cleaned, batch_size=settings.embedding_batch_size))
        return np.stack(embeddings).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single user query. Returns shape (dimension,)."""
        if not text or not text.strip():
            raise ValueError("embed_query() requires non-empty text")

        embedding = next(iter(self._model.query_embed([text.strip()])))
        return embedding.astype(np.float32)

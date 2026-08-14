"""
Embedding layer.

A thin, explicit wrapper around a Sentence-Transformers model. Its only
job is text -> vector, done consistently for both indexing and querying,
so the rest of the pipeline (FAISS store, retriever) never touches a
model directly. The model name is fully configurable via
config.settings.embedding_model -- swapping models means changing config,
not code.
"""
from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str | None = None, normalize: bool = True):
        """
        normalize=True L2-normalizes every embedding, which is what lets a
        FAISS inner-product index behave as cosine similarity downstream.
        """
        self.model_name = model_name or settings.embedding_model
        self.normalize = normalize
        logger.info("Loading embedding model: %s", self.model_name)
        self._model = SentenceTransformer(self.model_name)
        if hasattr(self._model, "get_embedding_dimension"):
            self._dimension = self._model.get_embedding_dimension()
        else:
            self._dimension = self._model.get_sentence_embedding_dimension()

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

        embeddings = self._model.encode(
            cleaned,
            batch_size=settings.embedding_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single user query. Returns shape (dimension,)."""
        if not text or not text.strip():
            raise ValueError("embed_query() requires non-empty text")

        embedding = self._model.encode(
            [text.strip()],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return embedding[0].astype(np.float32)

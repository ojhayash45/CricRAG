"""
Retriever: ties EmbeddingService + FAISSStore together into the
question -> chunks step of the RAG pipeline.

    user question
        -> query embedding
        -> FAISS search (top_k)
        -> score filtering (score_threshold)
        -> retrieved context

`search()` returns both the raw (unfiltered) and filtered results. The raw
list is what Streamlit's "Show retrieval details" debug view needs (Phase
23 of the spec) -- you want to see what was found even when it didn't
clear the relevance bar, to tell a bad-retrieval problem apart from a
too-strict threshold. The filtered list is what the RAG pipeline should
actually hand to the LLM.

score_threshold is a similarity score in [-1, 1] (cosine similarity, since
embeddings are L2-normalized). It is a configurable starting point
(config.settings.similarity_threshold), NOT an empirically calibrated
value -- see README/evaluation for how it should be tuned.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from config import settings
from src.embeddings.embedding_service import EmbeddingService
from src.vectorstore.faiss_store import FAISSStore, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    query: str
    top_k: int
    score_threshold: float
    raw_results: list[SearchResult] = field(default_factory=list)
    filtered_results: list[SearchResult] = field(default_factory=list)

    @property
    def has_sufficient_relevance(self) -> bool:
        return len(self.filtered_results) > 0


class Retriever:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: FAISSStore,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.default_top_k = top_k if top_k is not None else settings.top_k
        self.default_score_threshold = (
            score_threshold if score_threshold is not None else settings.similarity_threshold
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> RetrievalResult:
        if not query or not query.strip():
            raise ValueError("query must not be empty")

        k = top_k if top_k is not None else self.default_top_k
        threshold = score_threshold if score_threshold is not None else self.default_score_threshold

        query_embedding = self.embedding_service.embed_query(query)
        raw_results = self.vector_store.search(query_embedding, top_k=k)
        filtered_results = [r for r in raw_results if r.score >= threshold]

        logger.info(
            "Query %r: %d raw result(s), %d passed threshold %.3f",
            query, len(raw_results), len(filtered_results), threshold,
        )

        return RetrievalResult(
            query=query,
            top_k=k,
            score_threshold=threshold,
            raw_results=raw_results,
            filtered_results=filtered_results,
        )

from __future__ import annotations

import numpy as np
import pytest

from src.models.schemas import Chunk
from src.retrieval.retriever import Retriever
from src.vectorstore.faiss_store import FAISSStore

DIM = 4


def unit_vector(*components: float) -> np.ndarray:
    v = np.array(components, dtype=np.float32)
    return v / np.linalg.norm(v)


class StubEmbeddingService:
    """Deterministic stand-in for EmbeddingService -- no model load needed."""

    def __init__(self, query_vectors: dict[str, np.ndarray]):
        self._query_vectors = query_vectors
        self.dimension = DIM

    def embed_query(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            raise ValueError("embed_query() requires non-empty text")
        return self._query_vectors[text]


def make_chunk(chunk_id: str, **overrides) -> Chunk:
    defaults = dict(
        chunk_id=chunk_id,
        document_id="doc-1",
        law_number="36",
        law_title="Out Leg Before Wicket",
        section="36.1",
        edition="2017 Code",
        text=f"text for {chunk_id}",
        source_url="file://test",
        chunk_index=0,
        content_hash=f"hash-{chunk_id}",
    )
    defaults.update(overrides)
    return Chunk(**defaults)


@pytest.fixture
def populated_store() -> FAISSStore:
    store = FAISSStore(dimension=DIM)
    chunks = [
        make_chunk("close", section="36.1"),
        make_chunk("medium", section="36.2"),
        make_chunk("far", section="99.9"),
    ]
    embeddings = np.stack([
        unit_vector(1, 0, 0, 0),
        unit_vector(0.6, 0.8, 0, 0),
        unit_vector(0, 0, 0, 1),
    ])
    store.add(embeddings, chunks)
    return store


class TestRetriever:
    def test_returns_top_k_filtered_results(self, populated_store):
        embedder = StubEmbeddingService({"lbw?": unit_vector(1, 0, 0, 0)})
        retriever = Retriever(embedder, populated_store, top_k=2, score_threshold=0.5)

        result = retriever.search("lbw?")
        assert result.top_k == 2
        assert len(result.raw_results) == 2
        assert all(r.score >= 0.5 for r in result.filtered_results)
        assert result.filtered_results[0].chunk.chunk_id == "close"

    def test_threshold_filters_low_scoring_but_raw_keeps_them(self, populated_store):
        embedder = StubEmbeddingService({"lbw?": unit_vector(1, 0, 0, 0)})
        retriever = Retriever(embedder, populated_store, top_k=3, score_threshold=0.95)

        result = retriever.search("lbw?")
        assert len(result.raw_results) == 3
        assert len(result.filtered_results) == 1
        assert result.filtered_results[0].chunk.chunk_id == "close"
        assert result.has_sufficient_relevance is True

    def test_no_results_above_threshold_reports_insufficient_relevance(self, populated_store):
        embedder = StubEmbeddingService({"unrelated": unit_vector(0, 0, 1, 0)})
        retriever = Retriever(embedder, populated_store, top_k=3, score_threshold=0.99)

        result = retriever.search("unrelated")
        assert result.filtered_results == []
        assert result.has_sufficient_relevance is False

    def test_per_call_overrides_take_precedence_over_defaults(self, populated_store):
        embedder = StubEmbeddingService({"lbw?": unit_vector(1, 0, 0, 0)})
        retriever = Retriever(embedder, populated_store, top_k=1, score_threshold=0.99)

        result = retriever.search("lbw?", top_k=3, score_threshold=-1.0)
        assert result.top_k == 3
        assert result.score_threshold == -1.0
        assert len(result.filtered_results) == 3

    def test_empty_query_raises_value_error(self, populated_store):
        embedder = StubEmbeddingService({})
        retriever = Retriever(embedder, populated_store)
        with pytest.raises(ValueError):
            retriever.search("   ")

    def test_empty_index_returns_no_results_without_error(self):
        empty_store = FAISSStore(dimension=DIM)
        embedder = StubEmbeddingService({"lbw?": unit_vector(1, 0, 0, 0)})
        retriever = Retriever(embedder, empty_store)

        result = retriever.search("lbw?")
        assert result.raw_results == []
        assert result.filtered_results == []
        assert result.has_sufficient_relevance is False

    def test_defaults_come_from_settings_when_not_overridden(self, populated_store, monkeypatch):
        import config
        monkeypatch.setattr(config.settings, "top_k", 1)
        monkeypatch.setattr(config.settings, "similarity_threshold", -1.0)

        embedder = StubEmbeddingService({"lbw?": unit_vector(1, 0, 0, 0)})
        retriever = Retriever(embedder, populated_store)

        result = retriever.search("lbw?")
        assert result.top_k == 1
        assert result.score_threshold == -1.0

    def test_search_result_includes_chunk_and_score_fields(self, populated_store):
        embedder = StubEmbeddingService({"lbw?": unit_vector(1, 0, 0, 0)})
        retriever = Retriever(embedder, populated_store, top_k=1, score_threshold=-1.0)

        result = retriever.search("lbw?")
        top = result.filtered_results[0]
        assert top.chunk.chunk_id == "close"
        assert top.chunk.law_number == "36"
        assert top.chunk.law_title == "Out Leg Before Wicket"
        assert top.chunk.section == "36.1"
        assert top.chunk.source_url == "file://test"
        assert isinstance(top.score, float)

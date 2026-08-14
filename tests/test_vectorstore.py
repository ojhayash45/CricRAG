from __future__ import annotations

import numpy as np
import pytest

from src.models.schemas import Chunk
from src.vectorstore.faiss_store import FAISSStore

DIM = 4


def make_chunk(chunk_id: str, text: str, content_hash: str | None = None, **overrides) -> Chunk:
    defaults = dict(
        chunk_id=chunk_id,
        document_id="doc-1",
        law_number="36",
        law_title="Out Leg Before Wicket",
        section="36.1",
        edition="2017 Code",
        text=text,
        source_url="file://test",
        chunk_index=0,
        content_hash=content_hash or f"hash-{chunk_id}",
    )
    defaults.update(overrides)
    return Chunk(**defaults)


def unit_vector(*components: float) -> np.ndarray:
    v = np.array(components, dtype=np.float32)
    return v / np.linalg.norm(v)


class TestFAISSStoreAddAndSearch:
    def test_add_returns_count_and_updates_size(self):
        store = FAISSStore(dimension=DIM)
        chunks = [make_chunk("a", "text a"), make_chunk("b", "text b")]
        embeddings = np.stack([unit_vector(1, 0, 0, 0), unit_vector(0, 1, 0, 0)])
        added = store.add(embeddings, chunks)
        assert added == 2
        assert store.size == 2

    def test_search_returns_top_k_ordered_by_similarity(self):
        store = FAISSStore(dimension=DIM)
        chunks = [
            make_chunk("a", "close to query"),
            make_chunk("b", "somewhat related"),
            make_chunk("c", "unrelated"),
        ]
        embeddings = np.stack([
            unit_vector(1, 0, 0, 0),
            unit_vector(0.7, 0.7, 0, 0),
            unit_vector(0, 0, 1, 0),
        ])
        store.add(embeddings, chunks)

        query = unit_vector(1, 0, 0, 0)
        results = store.search(query, top_k=2)

        assert len(results) == 2
        assert results[0].chunk.chunk_id == "a"
        assert results[0].score >= results[1].score

    def test_search_result_carries_full_metadata(self):
        store = FAISSStore(dimension=DIM)
        chunk = make_chunk("a", "LBW rule text", section="36.1", law_number="36")
        store.add(np.stack([unit_vector(1, 0, 0, 0)]), [chunk])

        results = store.search(unit_vector(1, 0, 0, 0), top_k=1)
        result = results[0]
        assert result.chunk.section == "36.1"
        assert result.chunk.law_number == "36"
        assert result.chunk.source_url == "file://test"

    def test_search_on_empty_index_returns_empty_list(self):
        store = FAISSStore(dimension=DIM)
        results = store.search(unit_vector(1, 0, 0, 0), top_k=5)
        assert results == []

    def test_top_k_larger_than_index_size_does_not_error(self):
        store = FAISSStore(dimension=DIM)
        store.add(np.stack([unit_vector(1, 0, 0, 0)]), [make_chunk("a", "only one")])
        results = store.search(unit_vector(1, 0, 0, 0), top_k=10)
        assert len(results) == 1

    def test_add_rejects_mismatched_embedding_and_chunk_counts(self):
        store = FAISSStore(dimension=DIM)
        embeddings = np.stack([unit_vector(1, 0, 0, 0)])
        with pytest.raises(ValueError):
            store.add(embeddings, [make_chunk("a", "x"), make_chunk("b", "y")])

    def test_add_rejects_wrong_dimension(self):
        store = FAISSStore(dimension=DIM)
        embeddings = np.zeros((1, DIM + 1), dtype=np.float32)
        with pytest.raises(ValueError):
            store.add(embeddings, [make_chunk("a", "x")])

    def test_add_skips_duplicate_content_hash(self):
        store = FAISSStore(dimension=DIM)
        c1 = make_chunk("a", "same text", content_hash="dup-hash")
        c2 = make_chunk("b", "same text", content_hash="dup-hash")
        embeddings = np.stack([unit_vector(1, 0, 0, 0), unit_vector(0, 1, 0, 0)])
        added = store.add(embeddings, [c1, c2])
        assert added == 1
        assert store.size == 1


class TestFAISSStorePersistence:
    def test_save_and_load_round_trip(self, tmp_path):
        store = FAISSStore(dimension=DIM)
        chunks = [make_chunk("a", "text a"), make_chunk("b", "text b")]
        embeddings = np.stack([unit_vector(1, 0, 0, 0), unit_vector(0, 1, 0, 0)])
        store.add(embeddings, chunks)

        index_path = tmp_path / "index.faiss"
        metadata_path = tmp_path / "metadata.json"
        store.save(index_path, metadata_path)

        loaded = FAISSStore.load(index_path, metadata_path)
        assert loaded.size == 2
        assert loaded.dimension == DIM

        results = loaded.search(unit_vector(1, 0, 0, 0), top_k=1)
        assert results[0].chunk.chunk_id == "a"

    def test_load_missing_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FAISSStore.load(tmp_path / "missing.faiss", tmp_path / "missing.json")

    def test_loaded_store_still_deduplicates_on_add(self, tmp_path):
        store = FAISSStore(dimension=DIM)
        chunk = make_chunk("a", "text a", content_hash="dup-hash")
        store.add(np.stack([unit_vector(1, 0, 0, 0)]), [chunk])

        index_path = tmp_path / "index.faiss"
        metadata_path = tmp_path / "metadata.json"
        store.save(index_path, metadata_path)

        loaded = FAISSStore.load(index_path, metadata_path)
        duplicate = make_chunk("a-again", "text a", content_hash="dup-hash")
        added = loaded.add(np.stack([unit_vector(1, 0, 0, 0)]), [duplicate])
        assert added == 0
        assert loaded.size == 1

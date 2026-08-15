"""
Real integration tests against Postgres + pgvector -- skipped entirely if
DATABASE_URL isn't set (no local Postgres assumed). Uses an isolated,
disposable table name so this never touches real indexed data even if run
against the same database the app uses.
"""
from __future__ import annotations

import numpy as np
import pytest

from config import settings
from src.models.schemas import Chunk

DATABASE_URL = settings.database_url
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL not set -- skipping Postgres integration tests")

if DATABASE_URL:
    from src.vectorstore.postgres_store import PostgresVectorStore

TEST_TABLE = "chunks_pytest"
DIM = 4


def unit_vector(*components: float) -> np.ndarray:
    v = np.array(components, dtype=np.float32)
    return v / np.linalg.norm(v)


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
def store():
    s = PostgresVectorStore(table=TEST_TABLE)
    s.clear()  # start every test from a clean, guaranteed-not-to-exist table
    yield s
    s.clear()  # leave no trace afterwards


class TestPostgresVectorStore:
    def test_fresh_store_is_empty(self, store):
        assert store.size == 0
        assert store.search(unit_vector(1, 0, 0, 0), top_k=5) == []
        assert store.chunks == []

    def test_add_returns_count_and_updates_size(self, store):
        chunks = [make_chunk("a"), make_chunk("b")]
        embeddings = np.stack([unit_vector(1, 0, 0, 0), unit_vector(0, 1, 0, 0)])
        added = store.add(embeddings, chunks)
        assert added == 2
        assert store.size == 2

    def test_search_returns_top_k_ordered_by_similarity(self, store):
        chunks = [
            make_chunk("close", content_hash="h-close"),
            make_chunk("medium", content_hash="h-medium"),
            make_chunk("far", content_hash="h-far"),
        ]
        embeddings = np.stack([
            unit_vector(1, 0, 0, 0),
            unit_vector(0.7, 0.7, 0, 0),
            unit_vector(0, 0, 1, 0),
        ])
        store.add(embeddings, chunks)

        results = store.search(unit_vector(1, 0, 0, 0), top_k=2)
        assert len(results) == 2
        assert results[0].chunk.chunk_id == "close"
        assert results[0].score >= results[1].score

    def test_search_result_carries_full_metadata(self, store):
        chunk = make_chunk("a", section="36.1", law_number="36", source_url="file://test-source")
        store.add(np.stack([unit_vector(1, 0, 0, 0)]), [chunk])

        result = store.search(unit_vector(1, 0, 0, 0), top_k=1)[0]
        assert result.chunk.section == "36.1"
        assert result.chunk.law_number == "36"
        assert result.chunk.source_url == "file://test-source"

    def test_add_skips_duplicate_content_hash(self, store):
        c1 = make_chunk("a", content_hash="dup-hash")
        c2 = make_chunk("b", content_hash="dup-hash")
        embeddings = np.stack([unit_vector(1, 0, 0, 0), unit_vector(0, 1, 0, 0)])
        added = store.add(embeddings, [c1, c2])
        assert added == 1
        assert store.size == 1

    def test_add_is_idempotent_across_calls(self, store):
        chunk = make_chunk("a", content_hash="stable-hash")
        store.add(np.stack([unit_vector(1, 0, 0, 0)]), [chunk])
        added_again = store.add(np.stack([unit_vector(1, 0, 0, 0)]), [chunk])
        assert added_again == 0
        assert store.size == 1

    def test_clear_removes_all_data(self, store):
        store.add(np.stack([unit_vector(1, 0, 0, 0)]), [make_chunk("a")])
        assert store.size == 1
        store.clear()
        assert store.size == 0

    def test_chunks_property_returns_all_rows(self, store):
        chunks = [make_chunk("a", content_hash="h1"), make_chunk("b", content_hash="h2")]
        embeddings = np.stack([unit_vector(1, 0, 0, 0), unit_vector(0, 1, 0, 0)])
        store.add(embeddings, chunks)

        all_chunks = store.chunks
        assert {c.chunk_id for c in all_chunks} == {"a", "b"}

    def test_two_stores_with_different_tables_are_isolated(self):
        store_a = PostgresVectorStore(table=TEST_TABLE)
        store_b = PostgresVectorStore(table=f"{TEST_TABLE}_other")
        store_a.clear()
        store_b.clear()
        try:
            store_a.add(np.stack([unit_vector(1, 0, 0, 0)]), [make_chunk("only-in-a")])
            assert store_a.size == 1
            assert store_b.size == 0
        finally:
            store_a.clear()
            store_b.clear()

    def test_save_is_a_harmless_noop(self, store):
        store.add(np.stack([unit_vector(1, 0, 0, 0)]), [make_chunk("a")])
        store.save()  # should not raise
        assert store.size == 1

    def test_missing_database_url_raises_clear_error(self, monkeypatch):
        import config
        monkeypatch.setattr(config.settings, "database_url", None)
        with pytest.raises(ValueError):
            PostgresVectorStore(database_url=None)

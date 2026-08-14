from __future__ import annotations

import pytest

from src.chunking.chunker import SectionAwareChunker
from src.models.schemas import Document


def make_document(content: str, **overrides) -> Document:
    defaults = dict(
        title="Test Law",
        law_number="36",
        law_title="Out Leg Before Wicket",
        edition="2017 Code",
        source_url="file://test",
        content=content,
    )
    defaults.update(overrides)
    return Document(**defaults)


STRUCTURED_TEXT = (
    "Law 36 - Out Leg Before Wicket\n\n"
    "36.1 Out LBW\n"
    "A batter is out if the ball strikes the pad without a shot, subject to conditions.\n\n"
    "36.2 Interception outside off stump\n"
    "This does not apply if the batter was outside off stump playing a stroke.\n"
)


class TestSectionAwareChunker:
    def test_creates_one_chunk_per_small_section(self):
        chunker = SectionAwareChunker(chunk_size=650, chunk_overlap=80)
        doc = make_document(STRUCTURED_TEXT)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 2
        assert chunks[0].section == "36.1"
        assert chunks[1].section == "36.2"

    def test_preserves_metadata_on_every_chunk(self):
        chunker = SectionAwareChunker(chunk_size=650, chunk_overlap=80)
        doc = make_document(STRUCTURED_TEXT)
        chunks = chunker.chunk_document(doc)
        for chunk in chunks:
            assert chunk.document_id == doc.document_id
            assert chunk.law_number == "36"
            assert chunk.law_title == "Out Leg Before Wicket"
            assert chunk.edition == "2017 Code"
            assert chunk.source_url == "file://test"
            assert chunk.content_hash is not None

    def test_chunk_index_increments_across_sections(self):
        chunker = SectionAwareChunker(chunk_size=650, chunk_overlap=80)
        doc = make_document(STRUCTURED_TEXT)
        chunks = chunker.chunk_document(doc)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_handles_long_section_by_splitting(self):
        # 100 words, well beyond a small chunk_size, to force splitting.
        long_section_text = "Law 99 - Test\n\n99.1 Long section\n" + " ".join(
            f"word{i}" for i in range(100)
        )
        chunker = SectionAwareChunker(chunk_size=20, chunk_overlap=5)
        doc = make_document(long_section_text, law_number="99", law_title="Test")
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.section == "99.1"

    def test_overlap_is_respected_between_consecutive_pieces(self):
        long_section_text = "Law 99 - Test\n\n99.1 Long section\n" + " ".join(
            f"word{i}" for i in range(60)
        )
        chunker = SectionAwareChunker(chunk_size=20, chunk_overlap=5)
        doc = make_document(long_section_text, law_number="99", law_title="Test")
        chunks = chunker.chunk_document(doc)

        first_words = chunks[0].text.split()
        second_words = chunks[1].text.split()
        overlap_tail = set(first_words[-5:])
        overlap_head = set(second_words[:5])
        assert overlap_tail & overlap_head

    def test_rejects_overlap_not_smaller_than_chunk_size(self):
        with pytest.raises(ValueError):
            SectionAwareChunker(chunk_size=100, chunk_overlap=100)

    def test_unstructured_document_still_produces_a_chunk(self):
        doc = make_document("Just plain unstructured cricket rule text with no section markers.")
        chunker = SectionAwareChunker(chunk_size=650, chunk_overlap=80)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].section is None

    def test_chunk_documents_deduplicates_across_corpus(self):
        chunker = SectionAwareChunker(chunk_size=650, chunk_overlap=80)
        doc1 = make_document(STRUCTURED_TEXT, law_number="36")
        doc2 = make_document(STRUCTURED_TEXT, law_number="36")  # identical content
        chunks = chunker.chunk_documents([doc1, doc2])
        # Same text -> same hash -> deduplicated down to one document's worth.
        assert len(chunks) == 2

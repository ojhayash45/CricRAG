"""
Section-aware chunking.

Prefers natural document structure (Law -> Section -> rule text) over
blind fixed-size splitting:

    Law
    |-- Section
    |     `-- rule text  -> one chunk (if it fits)
    `-- Section (too large)
          `-- rule text  -> split into overlapping chunks, metadata preserved

"Tokens" here are approximated by whitespace word count, which avoids an
extra tokenizer dependency while staying explainable. CHUNK_SIZE /
CHUNK_OVERLAP are configurable (config.settings) and are starting points,
not a universally correct value -- see README limitations.
"""
from __future__ import annotations

import logging

from config import settings
from src.ingestion.cleaner import compute_content_hash, deduplicate_by_hash
from src.ingestion.parser import ParsedSection, parse_law_document
from src.models.schemas import Chunk, Document

logger = logging.getLogger(__name__)


class SectionAwareChunker:
    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    def chunk_document(self, document: Document) -> list[Chunk]:
        parsed = parse_law_document(document.content)
        law_number = parsed.law_number or document.law_number
        law_title = parsed.law_title or document.law_title

        chunks: list[Chunk] = []
        index = 0
        for section in parsed.sections:
            if not section.text.strip():
                continue
            words = section.text.split()
            if len(words) <= self.chunk_size:
                chunks.append(self._make_chunk(document, law_number, law_title, section, section.text, index))
                index += 1
            else:
                for piece_text in self._split_with_overlap(words):
                    chunks.append(self._make_chunk(document, law_number, law_title, section, piece_text, index))
                    index += 1

        logger.info(
            "Chunked document %s (Law %s) into %d chunk(s)",
            document.document_id, law_number, len(chunks),
        )
        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """Chunk many documents and drop exact-duplicate chunks across the corpus."""
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        deduped = deduplicate_by_hash(all_chunks, text_getter=lambda c: c.text)
        if len(deduped) != len(all_chunks):
            logger.info("Removed %d duplicate chunk(s)", len(all_chunks) - len(deduped))
        return deduped

    def _split_with_overlap(self, words: list[str]) -> list[str]:
        step = max(self.chunk_size - self.chunk_overlap, 1)
        pieces: list[str] = []
        start = 0
        while start < len(words):
            piece_words = words[start:start + self.chunk_size]
            if not piece_words:
                break
            pieces.append(" ".join(piece_words))
            if start + self.chunk_size >= len(words):
                break
            start += step
        return pieces

    @staticmethod
    def _make_chunk(
        document: Document,
        law_number: str | None,
        law_title: str | None,
        section: ParsedSection,
        text: str,
        index: int,
    ) -> Chunk:
        heading_prefix = f"{section.heading}\n" if section.heading else ""
        full_text = f"{heading_prefix}{text}".strip()
        return Chunk(
            document_id=document.document_id,
            law_number=law_number,
            law_title=law_title,
            section=section.section,
            edition=document.edition,
            text=full_text,
            source_url=document.source_url,
            chunk_index=index,
            content_hash=compute_content_hash(full_text),
        )

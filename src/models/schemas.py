"""
Core data models for the Cricket Laws RAG pipeline.

Document  -- one ingested source (a full Law, typically).
Chunk     -- one retrieval unit produced from a Document by the chunker.

Keeping these as explicit Pydantic models (rather than dicts) means every
stage of the pipeline validates its inputs/outputs and the shape of the
data is documented in one place.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class SourceType(str, Enum):
    """Where a Document originated from."""

    LOCAL_FILE = "local_file"


class Document(BaseModel):
    """A single ingested unit of source material -- usually one Law."""

    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    law_number: str | None = None
    law_title: str | None = None
    edition: str | None = None
    source_url: str | None = None
    source_type: SourceType = SourceType.LOCAL_FILE
    content: str
    content_hash: str | None = Field(
        default=None,
        description="Hash of normalized content, used for duplicate detection.",
    )
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("content")
    @classmethod
    def _content_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Document.content must not be empty")
        return v


class Chunk(BaseModel):
    """A single retrieval unit derived from a Document."""

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    law_number: str | None = None
    law_title: str | None = None
    section: str | None = None
    edition: str | None = None
    text: str
    source_url: str | None = None
    chunk_index: int = 0
    content_hash: str | None = Field(
        default=None,
        description="Hash of normalized chunk text, used for duplicate detection.",
    )

    @field_validator("text")
    @classmethod
    def _text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Chunk.text must not be empty")
        return v

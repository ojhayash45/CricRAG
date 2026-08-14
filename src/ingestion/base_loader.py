"""
Source adapter interface.

Every way of getting cricket-law text into the system implements
BaseSourceLoader so the rest of the pipeline (cleaner -> parser ->
chunker) never needs to know where a Document came from. Currently there
is one implementation, LocalDocumentLoader (.txt/.html/.pdf from
data/raw/), but the interface exists so another local-file-based source
could be added later without touching anything downstream.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.models.schemas import Document


class BaseSourceLoader(ABC):
    """Abstract interface for anything that can produce a list of Documents."""

    @abstractmethod
    def load(self) -> list[Document]:
        """Load and return raw (not-yet-cleaned) Documents."""
        raise NotImplementedError

"""
Local corpus loader.

Reads .txt, .html, and .pdf files from a directory (default: data/raw/)
and turns each one into one or more cleaned Documents. This is the only
ingestion path in this project -- see README's copyright section for why
there is no live-scraping alternative.

Small .txt/.html files are assumed to hold a single law (matching
parser.extract_law_header's "Law N - Title" shape) and become one
Document each. PDFs are assumed to potentially hold many laws in one file
(e.g. the official Laws of Cricket is ~180 pages covering 42 laws) --
parser.split_into_laws() finds each genuine law heading and this loader
turns every one of them into its own Document, all sharing the source
PDF's file:// URL. If no law structure is found in a PDF, it falls back
to indexing the whole file as one Document, same as a .txt file would be.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from config import settings
from src.ingestion.base_loader import BaseSourceLoader
from src.ingestion.cleaner import clean_text, compute_content_hash
from src.ingestion.parser import extract_edition, extract_law_header, split_into_laws
from src.models.schemas import Document, SourceType

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".txt", ".html", ".htm", ".pdf"}


def _title_case(text: str) -> str:
    """
    Like str.title(), but only capitalizes the first letter of each
    whitespace-separated word -- str.title() also capitalizes after
    apostrophes/hyphens, mangling possessives ("FIELDER'S" -> "Fielder'S").
    """
    def _cap_word(word: str) -> str:
        lower = word.lower()
        for i, ch in enumerate(lower):
            if ch.isalpha():
                return lower[:i] + ch.upper() + lower[i + 1:]
        return lower

    return "".join(_cap_word(w) if w.strip() else w for w in re.split(r"(\s+)", text))


def _extract_pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


class LocalDocumentLoader(BaseSourceLoader):
    """Loads an authorized/local corpus of cricket-law documents from disk."""

    def __init__(self, directory: Path | str | None = None):
        self.directory = Path(directory) if directory else settings.raw_data_dir

    def load(self) -> list[Document]:
        if not self.directory.exists():
            logger.warning("Local corpus directory does not exist: %s", self.directory)
            return []

        documents: list[Document] = []
        files = sorted(
            p for p in self.directory.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        )
        logger.info("Found %d candidate file(s) in %s", len(files), self.directory)

        for path in files:
            documents.extend(self._load_file(path))

        logger.info("Loaded %d document(s) from local corpus", len(documents))
        return documents

    def _load_file(self, path: Path) -> list[Document]:
        suffix = path.suffix.lower()
        is_pdf = suffix == ".pdf"

        try:
            if is_pdf:
                raw = _extract_pdf_text(path)
            else:
                raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.error("Failed to read %s: %s", path, exc)
            return []
        except Exception as exc:  # noqa: BLE001 - malformed/encrypted PDFs shouldn't crash ingestion
            logger.error("Failed to extract text from %s: %s", path, exc)
            return []

        cleaned = clean_text(raw, is_html=suffix in {".html", ".htm"})
        if not cleaned.strip():
            logger.warning("Skipping empty document after cleaning: %s", path)
            return []

        source_url = f"file://{path.resolve()}"

        if is_pdf:
            documents = self._documents_from_pdf_text(cleaned, source_url, path)
            if documents:
                return documents
            # No law structure detected -- fall through and index the whole PDF as one document.

        law_number, law_title = extract_law_header(cleaned)
        return [Document(
            title=law_title or path.stem,
            law_number=law_number,
            law_title=law_title,
            edition=None,
            source_url=source_url,
            source_type=SourceType.LOCAL_FILE,
            content=cleaned,
            content_hash=compute_content_hash(cleaned),
        )]

    def _documents_from_pdf_text(self, cleaned: str, source_url: str, path: Path) -> list[Document]:
        laws = split_into_laws(cleaned)
        if not laws:
            return []

        edition = extract_edition(cleaned[:5000])
        logger.info("Split %s into %d law(s)%s", path.name, len(laws), f" ({edition})" if edition else "")

        return [
            Document(
                title=f"Law {law.law_number} - {_title_case(law.law_title)}",
                law_number=law.law_number,
                law_title=_title_case(law.law_title),
                edition=edition,
                source_url=source_url,
                source_type=SourceType.LOCAL_FILE,
                content=f"Law {law.law_number} {law.law_title}\n\n{law.body_text}",
                content_hash=compute_content_hash(law.body_text),
            )
            for law in laws
        ]

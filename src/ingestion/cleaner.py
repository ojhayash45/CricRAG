"""
Text cleaning utilities used after raw extraction and before chunking.

Responsibilities:
  - strip HTML markup down to readable text
  - normalize whitespace (collapse blank lines / repeated spaces)
  - compute a stable content hash for duplicate detection
"""
from __future__ import annotations

import hashlib
import re

from bs4 import BeautifulSoup

_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def strip_html(html: str) -> str:
    """Remove HTML markup (scripts, styles, tags) and return plain text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    # get_text with a newline separator keeps block-level structure,
    # which section-aware chunking later relies on.
    text = soup.get_text(separator="\n")
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse repeated spaces/tabs and excess blank lines; trim each line."""
    lines = [_WHITESPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    collapsed = "\n".join(lines)
    collapsed = _BLANK_LINES_RE.sub("\n\n", collapsed)
    return collapsed.strip()


def compute_content_hash(text: str) -> str:
    """Hash of normalized text, used to detect exact-duplicate content."""
    normalized = normalize_whitespace(text).lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def clean_text(raw: str, *, is_html: bool = False) -> str:
    """Full cleaning pipeline: optional HTML stripping + whitespace normalization."""
    text = strip_html(raw) if is_html else raw
    return normalize_whitespace(text)


def deduplicate_by_hash(items: list, text_getter=lambda x: x) -> list:
    """
    Remove items whose normalized content hash has already been seen.
    Preserves the first occurrence and original order.
    """
    seen: set[str] = set()
    result = []
    for item in items:
        h = compute_content_hash(text_getter(item))
        if h in seen:
            continue
        seen.add(h)
        result.append(item)
    return result

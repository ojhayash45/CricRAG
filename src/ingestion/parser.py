"""
Metadata + structure extraction for Laws-of-Cricket-style text.

Expected input shape for a single-law document (after cleaning), e.g.:

    Law 36 - Out Leg Before Wicket

    36.1 Out LBW
    <rule text...>

    36.2 Player treated as being struck by bat
    <rule text...>

`parse_law_document` handles that shape (used for small, hand-written
single-law files such as the test fixtures). `split_into_laws` handles the
other real-world shape this project actually ingests: a single large PDF
containing many laws (e.g. the official Laws of Cricket, ~180 pages,
42 laws in one file) -- see its docstring for how genuine headings are
told apart from the table of contents.

This module is intentionally regex-based and simple: the goal is an
explainable parser, not a general-purpose document understanding system.
If a source doesn't follow either pattern, both functions degrade
gracefully (one unnumbered section / an empty law list) rather than
raising.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_LAW_HEADER_RE = re.compile(
    r"^\s*Law\s+(?P<number>\d+[A-Za-z]?)\s*[-–—:]\s*(?P<title>.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Matches section markers like "36.1" or "2.1" at the start of a line,
# optionally followed by a heading on the same line. Deliberately stops at
# two levels: deeper enumeration such as "36.1.1" (a sub-point *within*
# section 36.1) is intentionally NOT treated as its own boundary -- it
# stays as body text inside its parent section's chunk. Splitting at every
# sub-point would fragment a single dismissal's conditions (e.g. LBW's five
# clauses) into five near-meaningless one-line chunks.
_SECTION_RE = re.compile(
    r"^\s*(?P<section>\d+\.\d+)(?!\.\d)[ \t]*(?P<heading>[^\n]*)$",
    re.MULTILINE,
)

# Genuine top-level headings in a real Laws/Playing-Conditions PDF are
# printed in full caps on their own line ("LAW 1 THE PLAYERS", or just
# "2 THE UMPIRES" for documents that don't use the word "Law"). An
# optional leading run of digits absorbs a page-footer number that PDF
# text extraction sometimes fuses onto the following line.
_MULTI_LAW_HEADER_RE = re.compile(
    r"^(?:\d{1,4}\s+)?(?:LAW\s+)?(?P<number>\d{1,2}[A-Za-z]?)\s+(?P<title>[A-Z][A-Z0-9 ,;'’\-]{2,70})\s*$",
    re.MULTILINE,
)

# A line that ends in a bare 1-4 digit number is almost always a table-of-
# contents or index entry (title ... page-number), never real prose.
_TRAILING_PAGE_NUMBER_RE = re.compile(r"\s\d{1,4}\s*$")

# e.g. "(2017 Code 4th Edition - 2026)" on the Laws of Cricket's title page.
_EDITION_RE = re.compile(r"\((\d{4}\s+Code[^)]*)\)", re.IGNORECASE)


@dataclass
class ParsedSection:
    section: str | None
    heading: str | None
    text: str


@dataclass
class ParsedLaw:
    law_number: str | None
    law_title: str | None
    sections: list[ParsedSection]


@dataclass
class SplitLaw:
    law_number: str
    law_title: str
    body_text: str


def extract_law_header(text: str) -> tuple[str | None, str | None]:
    """Return (law_number, law_title) from the first 'Law N - Title' line found."""
    match = _LAW_HEADER_RE.search(text)
    if not match:
        return None, None
    return match.group("number"), match.group("title").strip()


def extract_edition(text: str) -> str | None:
    """Return an edition/version string like '2017 Code 4th Edition - 2026' if present."""
    match = _EDITION_RE.search(text)
    return match.group(1).strip() if match else None


def _looks_like_listing(span_text: str, sample_lines: int = 12, threshold: float = 0.35) -> bool:
    """
    Real law bodies are prose; a table of contents (or a separate, more
    detailed index -- official Law/rules PDFs sometimes have both) lists
    each heading followed by a page number instead. Checking whether the
    text right after a candidate heading is mostly "page-number-shaped"
    lines is what lets split_into_laws() tell a genuine heading apart from
    a listing that merely repeats the same heading text.
    """
    lines = [line for line in span_text.splitlines() if line.strip()][:sample_lines]
    if not lines:
        return False
    hits = sum(1 for line in lines if _TRAILING_PAGE_NUMBER_RE.search(line))
    return (hits / len(lines)) > threshold


def split_into_laws(text: str) -> list[SplitLaw]:
    """
    Split a large, multi-law document's text (typically extracted from a
    real PDF spanning many laws in one file) into one SplitLaw per genuine
    heading found.

    Real official PDFs of this kind typically also contain a contents page
    and sometimes a separate detailed index, both of which repeat the same
    heading text -- _looks_like_listing filters those out so only actual
    body content is split into laws. Returns an empty list if no genuine
    headings are found at all (caller should fall back to treating the
    whole text as one undivided document).
    """
    matches = list(_MULTI_LAW_HEADER_RE.finditer(text))
    laws: list[SplitLaw] = []

    for i, m in enumerate(matches):
        title = m.group("title").strip()
        if "LAW" in title.split():
            # A page-footer number fused onto the next line by PDF text
            # extraction (e.g. "24 LAW 6 THE PITCH") -- not a real title.
            continue

        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]
        if _looks_like_listing(body):
            continue

        laws.append(SplitLaw(law_number=m.group("number"), law_title=title, body_text=body.strip()))

    return laws


def parse_law_document(text: str) -> ParsedLaw:
    """
    Split a cleaned, single-law document's text into (law_number, law_title, sections).

    If no section markers (e.g. "36.1") are found, the whole document is
    returned as a single section with section=None so downstream code can
    still chunk it safely.
    """
    law_number, law_title = extract_law_header(text)

    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        body = text
        if law_title:
            # Drop the "Law N - Title" header line from the body if present.
            body = _LAW_HEADER_RE.sub("", text, count=1).strip()
        return ParsedLaw(law_number=law_number, law_title=law_title, sections=[
            ParsedSection(section=None, heading=None, text=body.strip())
        ] if body.strip() else [])

    sections: list[ParsedSection] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_id = m.group("section")
        heading = m.group("heading").strip() or None
        body = text[start:end].strip()
        sections.append(ParsedSection(section=section_id, heading=heading, text=body))

    return ParsedLaw(law_number=law_number, law_title=law_title, sections=sections)

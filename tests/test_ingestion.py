from __future__ import annotations

from pathlib import Path

from src.ingestion.cleaner import (
    clean_text,
    compute_content_hash,
    deduplicate_by_hash,
    normalize_whitespace,
    strip_html,
)
from src.ingestion.document_loader import LocalDocumentLoader
from src.ingestion.parser import extract_edition, extract_law_header, parse_law_document, split_into_laws

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Cleaner
# ---------------------------------------------------------------------------

class TestCleaner:
    def test_strip_html_removes_tags_and_scripts(self):
        html = "<html><body><script>evil()</script><p>Hello <b>world</b></p></body></html>"
        text = strip_html(html)
        assert "evil()" not in text
        assert "<b>" not in text
        assert "Hello" in text and "world" in text

    def test_normalize_whitespace_collapses_spaces_and_blank_lines(self):
        messy = "Law   36  -  LBW\n\n\n\n36.1   Out LBW\n\n\n\ntext here"
        result = normalize_whitespace(messy)
        assert "   " not in result
        assert "\n\n\n" not in result

    def test_clean_text_preserves_rule_text_content(self):
        html = "<p>36.1 Out LBW</p><p>The batter is out if the ball strikes the pad.</p>"
        cleaned = clean_text(html, is_html=True)
        assert "36.1 Out LBW" in cleaned
        assert "strikes the pad" in cleaned

    def test_compute_content_hash_stable_and_case_insensitive(self):
        h1 = compute_content_hash("Hello World")
        h2 = compute_content_hash("hello   world")
        assert h1 == h2

    def test_deduplicate_by_hash_removes_exact_duplicates(self):
        items = ["same text", "SAME   TEXT", "different text"]
        deduped = deduplicate_by_hash(items)
        assert len(deduped) == 2
        assert deduped[0] == "same text"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class TestParser:
    def test_extract_law_header(self):
        text = "Law 36 - Out Leg Before Wicket\n\n36.1 Out LBW\ntext"
        number, title = extract_law_header(text)
        assert number == "36"
        assert title == "Out Leg Before Wicket"

    def test_extract_law_header_missing_returns_none(self):
        number, title = extract_law_header("no header here")
        assert number is None
        assert title is None

    def test_parse_law_document_splits_sections(self):
        text = (
            "Law 21 - No Ball\n\n"
            "21.1 Front foot no ball\n"
            "Some text about front foot.\n\n"
            "21.2 Back foot no ball\n"
            "Some text about back foot."
        )
        parsed = parse_law_document(text)
        assert parsed.law_number == "21"
        assert parsed.law_title == "No Ball"
        assert len(parsed.sections) == 2
        assert parsed.sections[0].section == "21.1"
        assert "front foot" in parsed.sections[0].text
        assert parsed.sections[1].section == "21.2"
        assert "back foot" in parsed.sections[1].text

    def test_parse_law_document_without_sections_returns_single_section(self):
        text = "Law 1 - The Players\nJust some unstructured text about players."
        parsed = parse_law_document(text)
        assert len(parsed.sections) == 1
        assert parsed.sections[0].section is None
        assert "unstructured text" in parsed.sections[0].text

    def test_parse_law_document_does_not_split_at_third_level_subpoints(self):
        # Real Laws-of-Cricket text enumerates conditions like this under a single
        # section (e.g. LBW's five clauses) -- these must stay part of 36.1's
        # chunk, not fragment into five near-meaningless one-line chunks.
        text = (
            "Law 36 - Leg Before Wicket\n\n"
            "36.1 Out LBW\n"
            "The striker is out LBW if all of the following apply.\n"
            "36.1.1 The bowler delivers a ball, not being a No ball.\n"
            "36.1.2 The ball first lands in line between wicket and wicket.\n"
            "36.2 Off side of wicket\n"
            "The off side is determined by the striker's stance."
        )
        parsed = parse_law_document(text)
        assert [s.section for s in parsed.sections] == ["36.1", "36.2"]
        assert "36.1.1" in parsed.sections[0].text
        assert "36.1.2" in parsed.sections[0].text


# ---------------------------------------------------------------------------
# split_into_laws (multi-law PDF documents)
# ---------------------------------------------------------------------------

# Mimics the real shape of an official Laws-of-Cricket-style PDF: a brief,
# mixed-case contents page (never matches -- title case is excluded on
# purpose), the real all-caps content, and a detailed all-caps index that
# repeats the same headings with trailing page numbers.
_MULTI_LAW_SAMPLE = (
    "CONTENTS\n"
    "Law 1 The players 9\n"
    "Law 2 The umpires 10\n"
    "\n"
    "LAW 1 THE PLAYERS\n"
    "1.1 Number of players\n"
    "A match is played between two sides of eleven players each.\n"
    "1.2 Nomination of players\n"
    "Each captain shall nominate their players before the toss.\n"
    "\n"
    "LAW 2 THE UMPIRES\n"
    "2.1 Appointment and attendance\n"
    "Two umpires shall control the game as required by these laws.\n"
    "2.2 Change of umpire\n"
    "An umpire shall not be changed during a match.\n"
    "\n"
    "INDEX\n"
    "LAW 1  THE PLAYERS\n"
    "1.1  Number of players 9\n"
    "1.2  Nomination of players 9\n"
    "LAW 2  THE UMPIRES\n"
    "2.1  Appointment and attendance 10\n"
    "2.2  Change of umpire 10\n"
)


class TestSplitIntoLaws:
    def test_finds_genuine_headings_and_ignores_toc_and_index_duplicates(self):
        laws = split_into_laws(_MULTI_LAW_SAMPLE)
        assert [(law.law_number, law.law_title) for law in laws] == [
            ("1", "THE PLAYERS"),
            ("2", "THE UMPIRES"),
        ]

    def test_body_text_contains_real_prose_not_index_entries(self):
        laws = split_into_laws(_MULTI_LAW_SAMPLE)
        players = next(law for law in laws if law.law_number == "1")
        assert "eleven players" in players.body_text
        assert "1.1  Number of players 9" not in players.body_text

    def test_absorbs_page_footer_number_fused_onto_a_heading_line(self):
        # PDF text extraction sometimes leaves a page-footer number directly
        # before the next page's heading with only a newline between them
        # (no blank line) -- e.g. "...24\nLAW 6 THE PITCH". The stray "24"
        # must be absorbed, not misread as the law number or folded into the title.
        text = (
            "...are not recognised in the Laws.\n"
            "24\n"
            "LAW 6 THE PITCH\n"
            "The pitch is a rectangular area of the ground 22 yards long "
            "and 10 feet wide, used for play.\n"
        )
        laws = split_into_laws(text)
        assert len(laws) == 1
        assert laws[0].law_number == "6"
        assert laws[0].law_title == "THE PITCH"
        assert "rectangular area" in laws[0].body_text

    def test_no_headings_returns_empty_list(self):
        assert split_into_laws("Just some unstructured plain text with no headings.") == []

    def test_extract_edition_from_title_page_text(self):
        text = "OFFICIAL\nTHE LAWS OF CRICKET\n(2017 Code 4th Edition - 2026)\n"
        assert extract_edition(text) == "2017 Code 4th Edition - 2026"

    def test_extract_edition_missing_returns_none(self):
        assert extract_edition("No edition information here.") is None


# ---------------------------------------------------------------------------
# LocalDocumentLoader
# ---------------------------------------------------------------------------

class TestLocalDocumentLoader:
    def test_loads_txt_and_html_files(self):
        loader = LocalDocumentLoader(FIXTURES)
        docs = loader.load()
        assert len(docs) == 2
        law_numbers = {d.law_number for d in docs}
        assert law_numbers == {"36", "21"}

    def test_document_has_source_url_and_hash(self):
        loader = LocalDocumentLoader(FIXTURES)
        docs = loader.load()
        for doc in docs:
            assert doc.source_url is not None and doc.source_url.startswith("file://")
            assert doc.content_hash is not None

    def test_missing_directory_returns_empty_list(self, tmp_path):
        loader = LocalDocumentLoader(tmp_path / "does_not_exist")
        assert loader.load() == []

    def test_html_content_has_no_markup(self):
        loader = LocalDocumentLoader(FIXTURES)
        docs = loader.load()
        no_ball_doc = next(d for d in docs if d.law_number == "21")
        assert "<p>" not in no_ball_doc.content
        assert "console.log" not in no_ball_doc.content


# ---------------------------------------------------------------------------
# LocalDocumentLoader + PDFs (pypdf extraction is mocked -- no real PDF file
# needed; split_into_laws itself is already covered directly above)
# ---------------------------------------------------------------------------

class TestLocalDocumentLoaderPdf:
    def test_multi_law_pdf_becomes_one_document_per_law(self, tmp_path, monkeypatch):
        import src.ingestion.document_loader as document_loader

        pdf_path = tmp_path / "laws.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake bytes, never actually parsed")
        monkeypatch.setattr(document_loader, "_extract_pdf_text", lambda path: _MULTI_LAW_SAMPLE)

        docs = LocalDocumentLoader(tmp_path).load()
        assert len(docs) == 2
        assert {d.law_number for d in docs} == {"1", "2"}
        assert all(d.source_url == f"file://{pdf_path.resolve()}" for d in docs)

    def test_law_title_is_title_cased_not_shouting(self, tmp_path, monkeypatch):
        import src.ingestion.document_loader as document_loader

        (tmp_path / "laws.pdf").write_bytes(b"%PDF-1.4 fake")
        monkeypatch.setattr(document_loader, "_extract_pdf_text", lambda path: _MULTI_LAW_SAMPLE)

        docs = LocalDocumentLoader(tmp_path).load()
        umpires = next(d for d in docs if d.law_number == "2")
        assert umpires.law_title == "The Umpires"

    def test_pdf_without_law_structure_falls_back_to_single_document(self, tmp_path, monkeypatch):
        import src.ingestion.document_loader as document_loader

        (tmp_path / "notes.pdf").write_bytes(b"%PDF-1.4 fake")
        monkeypatch.setattr(
            document_loader, "_extract_pdf_text",
            lambda path: "Just some unstructured plain text with no law headings at all.",
        )

        docs = LocalDocumentLoader(tmp_path).load()
        assert len(docs) == 1
        assert docs[0].law_number is None
        assert "unstructured plain text" in docs[0].content

    def test_pdf_extraction_failure_is_skipped_not_raised(self, tmp_path, monkeypatch):
        import src.ingestion.document_loader as document_loader

        (tmp_path / "corrupt.pdf").write_bytes(b"not a real pdf")

        def _raise(path):
            raise ValueError("simulated corrupt PDF")

        monkeypatch.setattr(document_loader, "_extract_pdf_text", _raise)
        assert LocalDocumentLoader(tmp_path).load() == []

    def test_edition_extracted_and_shared_across_laws_from_same_pdf(self, tmp_path, monkeypatch):
        import src.ingestion.document_loader as document_loader

        (tmp_path / "laws.pdf").write_bytes(b"%PDF-1.4 fake")
        text_with_edition = "(2017 Code 4th Edition - 2026)\n\n" + _MULTI_LAW_SAMPLE
        monkeypatch.setattr(document_loader, "_extract_pdf_text", lambda path: text_with_edition)

        docs = LocalDocumentLoader(tmp_path).load()
        assert len(docs) == 2
        assert all(d.edition == "2017 Code 4th Edition - 2026" for d in docs)

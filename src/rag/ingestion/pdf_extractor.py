from __future__ import annotations

import asyncio
import bisect
import difflib
import io
import math
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

try:
    import pandas as pd
except ImportError:  # pragma: no cover - dependency presence is environment-specific
    pd = None

try:
    import pdfplumber as _pdfplumber
except ImportError:  # pragma: no cover - dependency presence is environment-specific
    _pdfplumber = None

try:
    from pdfplumber.utils.text import LIGATURES as _PDFPLUMBER_LIGATURES
except ImportError:  # pragma: no cover - dependency presence is environment-specific
    _PDFPLUMBER_LIGATURES = {}
_LIGATURES = str.maketrans(_PDFPLUMBER_LIGATURES)
_SPACED_LETTERS = re.compile(r"(?<!\S)(?:[^\W\d_] ){5,}[^\W\d_](?!\S)")  # 6+ single letters in a row: "e n s u r i"

from src.rag.reversed_text import turn_reversed_runs

try:
    from rapidfuzz import fuzz as _rapidfuzz
except ImportError:  # pragma: no cover - dependency presence is environment-specific
    _rapidfuzz = None


@dataclass(slots=True)
class PDFSegment:
    page_number: int
    segment_type: str
    text: str
    y0: float
    x0: float
    # Right and bottom edge of a layout text box (0.0 elsewhere): _join_lone_labels measures gaps between boxes.
    x1: float = 0.0
    y1: float = 0.0
    # Font size (median of the characters) where a layout piece starts and where it ends; 0.0 when not known.
    # _join_paragraphs reads a jump in size as a heading boundary.
    size: float = 0.0
    end_size: float = 0.0


# A layout text box this short (in words), with a letter and no digit, is a lone label: a chart legend entry
# ("Sukkur"), a map name, a word set apart by justified spacing ("thousand"), a heading. See _join_lone_labels.
LABEL_MAX_WORDS = 3

# _join_paragraphs: text boxes are joined into pieces of about this many characters (the default chunk size,
# settings.rag_chunk_size), so a passage holds whole sentences; the fixed-window chunker then windows longer pieces.
PARAGRAPH_MAX_CHARS = 800
# A box this much bigger (font size ratio) than the text before it starts a new piece: a heading.
SIZE_JUMP = 1.15
# Vertical gap, in multiples of the font size: up to LINE_GAP the next box is the same paragraph's next line (joined
# with a space); up to SECTION_GAP it is the next paragraph of the same stretch of text (joined with a line break, if
# the piece has room); further away it is somewhere else on the page (a footer, a figure) and starts a new piece.
LINE_GAP = 1.2
SECTION_GAP = 3.0
FALLBACK_SIZE = 11.0  # when the font size is not known (test doubles without characters)
HEADING_MAX_CHARS = 200  # a piece this short, all in bigger type than the next box, is a heading: it takes that box
PARAGRAPH_HARD_CAP = 3  # one paragraph's lines stay together up to this many times max_chars

# _running_headers: a text box wholly inside the top or bottom HEADER_BAND of its page (a share of the page height)
# whose text, with every run of digits read as one "#", sits at the same height (within HEADER_Y_TOLERANCE of the
# page height) on at least HEADER_MIN_SHARE of the document's pages and on at least HEADER_MIN_PAGES pages is a
# running header or footer ("The State of Children in Pakistan 2024", "Rights of Child | 87", a page number alone).
HEADER_BAND = 0.12
HEADER_MIN_SHARE = 0.3
HEADER_MIN_PAGES = 3
HEADER_Y_TOLERANCE = 0.01
_DIGIT_RUN = re.compile(r"\d+")
_BARE_NUMBER = re.compile(r"[-+]?(\d[\d,]*(\.\d+)?|\.\d+)%?")  # a cell that is only a number: data, not a header
# A cell holding an amount: a number grouped with commas ("6,500", "12,400 students") or a number after a currency
# ("Rs 6500", "PKR 4,850", "$12"): data, not a column name (review 20 Nov: "| Nursery | Rs 6,500 |" became the header).
_AMOUNT = re.compile(r"\d{1,3}(,\d{3})+|(\b(rs|pkr|usd|inr|aed|sar)\.?|[$£€₹])\s*\d", re.IGNORECASE)


@dataclass(slots=True)
class _PageRead:
    """One page read but not yet joined: the text boxes (layout path) or line blocks (word path) and the tables.
    path is "layout", "words" or "rtl" (an Urdu page: one segment, no layout, never a running-header candidate)."""
    page_number: int
    height: float
    path: str
    boxes: list[PDFSegment]
    tables: list[PDFSegment]


@dataclass(slots=True)
class PDFExtractionResult:
    pages_total: int
    pages_ingested: int
    skipped_pages: list[int]
    warnings: list[str]
    segments: list[PDFSegment]
    # The running headers and footers left out of segments (see _running_headers). They are still text on the page:
    # the Add documents check for picture-only pages counts them, as it did before they were left out.
    running_headers: list[PDFSegment] = field(default_factory=list)


class PDFExtractor(Protocol):
    async def extract(
        self,
        *,
        pdf_bytes: bytes,
        max_pages: int,
    ) -> PDFExtractionResult: ...


class PDFPlumberExtractor:
    def __init__(self, *, dedupe_threshold: int = 96, paragraph_chars: int = PARAGRAPH_MAX_CHARS) -> None:
        self._dedupe_threshold = dedupe_threshold
        self._paragraph_chars = paragraph_chars

    async def extract(
        self,
        *,
        pdf_bytes: bytes,
        max_pages: int,
        on_page=None,
    ) -> PDFExtractionResult:
        return await asyncio.to_thread(
            self._extract_sync,
            pdf_bytes=pdf_bytes,
            max_pages=max_pages,
            on_page=on_page,
        )

    def _extract_sync(
        self,
        *,
        pdf_bytes: bytes,
        max_pages: int,
        on_page=None,
    ) -> PDFExtractionResult:
        """on_page(done, total), when given, is told the pages read so far ((0, total) first, then after each page),
        so the Add documents page can say "Reading page 40 of 182". It runs in this (worker) thread."""
        if _pdfplumber is None:
            raise RuntimeError("pdfplumber is not installed. Install it to use PDF ingestion.")
        if not pdf_bytes:
            raise ValueError("Uploaded PDF is empty.")

        # laparams: pdfminer's layout analysis (pdfplumber is built on it) groups the words into text boxes in reading
        # order. Without it, words at the same height were joined across the page, so a two-column report read
        # "...toilet facilities. / where schools in Punjab and KP ... young girls who" and its facts were not found
        # (public eval, 3 Nov: Q9, Q10). all_texts: text inside a figure or Form XObject (chart labels, a running
        # header, a pdfjam page body) gets text boxes too; without it the layout path lost that text (review 3 Nov).
        with _pdfplumber.open(io.BytesIO(pdf_bytes), laparams={"all_texts": True}) as pdf:
            pages = list(pdf.pages)
            pages_total = len(pages)
            if pages_total == 0:
                raise ValueError("PDF has no pages.")
            if pages_total > max_pages:
                raise ValueError(
                    f"PDF has {pages_total} pages; maximum allowed is {max_pages}."
                )

            all_segments: list[PDFSegment] = []
            skipped_pages: list[int] = []
            warnings: list[str] = []
            pages_ingested = 0
            if on_page is not None:
                on_page(0, pages_total)

            # Every page is read first, so running headers and footers can be found across the whole document and
            # left out before a page's boxes are joined into pieces.
            reads: list[_PageRead | Exception] = []
            for page_number, page in enumerate(pages, start=1):
                try:
                    reads.append(self._read_page(page=page, page_number=page_number))
                except Exception as exc:
                    reads.append(exc)
                finally:
                    if on_page is not None:
                        on_page(page_number, pages_total)
            running = self._running_headers([r for r in reads if isinstance(r, _PageRead)], pages_total=pages_total)
            running_headers: list[PDFSegment] = []

            for page_number, read in enumerate(reads, start=1):
                try:
                    if isinstance(read, Exception):
                        raise read
                    page_segments = self._finish_page(read, drop=running)
                except Exception as exc:
                    skipped_pages.append(page_number)
                    warnings.append(f"Page {page_number}: extraction failed ({exc})")
                    continue

                running_headers.extend(b for i, b in enumerate(read.boxes) if (page_number, i) in running)
                if not page_segments:
                    skipped_pages.append(page_number)
                    warnings.append(f"Page {page_number}: no extractable text or tables.")
                    continue

                pages_ingested += 1
                all_segments.extend(page_segments)

            return PDFExtractionResult(
                pages_total=pages_total,
                pages_ingested=pages_ingested,
                skipped_pages=skipped_pages,
                warnings=warnings,
                segments=all_segments,
                running_headers=running_headers,
            )

    @staticmethod
    def _mostly_arabic_script(page: Any) -> bool:
        """True when most letters on the page are Arabic script (Urdu, Arabic, Persian)."""
        letters = [c.get("text", "") for c in (getattr(page, "chars", None) or []) if str(c.get("text", "")).isalpha()]
        arabic = sum(1 for t in letters if any("\u0600" <= ch <= "\u06ff" or "\ufb50" <= ch <= "\ufeff" for ch in t))
        return bool(letters) and arabic / len(letters) > 0.5

    def _extract_page_segments(self, *, page: Any, page_number: int) -> list[PDFSegment]:
        """One page on its own (no running-header removal: that needs the other pages)."""
        return self._finish_page(self._read_page(page=page, page_number=page_number), drop=frozenset())

    def _finish_page(self, read: _PageRead, *, drop: frozenset[tuple[int, int]] | set[tuple[int, int]]
                     ) -> list[PDFSegment]:
        """A read page's segments: its boxes minus those in drop ((page_number, box index): running headers and
        footers), then joined as the page's path joins them. Nothing else changes."""
        boxes = [b for i, b in enumerate(read.boxes) if (read.page_number, i) not in drop]
        if read.path == "rtl":
            return boxes
        if read.path == "layout":
            # Two text boxes are never near-duplicates of each other here: repeated chart labels ("8%"), footnotes
            # ("Ibid.") and "3 Barón et al" / "4 Barón et al" are separate boxes, and all of them were dropped but one.
            pieces = self._join_paragraphs(self._join_lone_labels(self._insert_tables(boxes, read.tables)),
                                           max_chars=self._paragraph_chars)
            return self._dedupe_adjacent_segments(pieces, keep_text_repeats=True)
        # The same lone-label join as the layout path, on the blocks and tables in top-to-bottom order.
        merged = self._join_lone_labels(sorted(
            [*boxes, *read.tables],
            key=lambda segment: (segment.y0, segment.x0),
        ))
        return self._dedupe_adjacent_segments(merged)

    @staticmethod
    def _header_key(text: str) -> str:
        """A box's text with spacing collapsed and every run of digits as "#": "Rights of Child | 87" and "Rights of
        Child | 88" are the same running footer, and every page number alone is "#"."""
        return _DIGIT_RUN.sub("#", " ".join(text.split()))

    @classmethod
    def _running_headers(cls, reads: list[_PageRead], *, pages_total: int) -> set[tuple[int, int]]:
        """(page_number, box index) of every running header and footer in the document, to be left out of the passages.

        4 Nov: after the paragraph join most of the remaining short passages (828 under 100 characters over 9 public
        PDFs) were running headers and footers repeated on every page ("The State of Children in Pakistan 2024",
        "...Rights of Child | 87"); they drew short questions and added noise. Decided by structure only, no word
        lists: a text box (layout path) or block of lines (word path) wholly inside the top or bottom HEADER_BAND of
        its page, whose _header_key is found in the same band at the same height (HEADER_Y_TOLERANCE) on at least
        HEADER_MIN_SHARE of the pages and at least HEADER_MIN_PAGES pages; its first occurrence (earliest page) stays,
        unless it is a page number alone (no letter). A box on only a few pages (a chapter
        heading, a table caption) is never dropped, nor anything in the middle of a page, a table, or an Urdu page
        (read as one piece without layout), nor all of a page's text. Documents under HEADER_MIN_PAGES pages are left as
        they are."""
        need = max(HEADER_MIN_PAGES, math.ceil(HEADER_MIN_SHARE * pages_total))
        if pages_total < HEADER_MIN_PAGES:
            return set()
        # (band, key) -> [(height on the page as a share of the page height, page_number, box index)]
        found: dict[tuple[str, str], list[tuple[float, int, int]]] = {}
        for read in reads:
            if read.path == "rtl" or read.height <= 0:
                continue
            for i, box in enumerate(read.boxes):
                if box.segment_type != "text" or box.y1 <= box.y0:
                    continue
                if box.y1 <= HEADER_BAND * read.height:
                    band, y = "top", box.y0 / read.height
                elif box.y0 >= (1 - HEADER_BAND) * read.height:
                    band, y = "bottom", box.y1 / read.height
                else:
                    continue
                found.setdefault((band, cls._header_key(box.text)), []).append((y, read.page_number, i))

        drop: set[tuple[int, int]] = set()
        for occurrences in found.values():
            if len({page for _, page, _ in occurrences}) < need:
                continue
            occurrences.sort()
            heights = [y for y, _, _ in occurrences]
            for y, page, i in occurrences:  # pages with the same text at (about) the same height as this one
                lo = bisect.bisect_left(heights, y - HEADER_Y_TOLERANCE)
                hi = bisect.bisect_right(heights, y + HEADER_Y_TOLERANCE)
                if len({p for _, p, _ in occurrences[lo:hi]}) >= need:
                    drop.add((page, i))
        # The first one is kept: the earliest occurrence of each header or footer with a letter in it stays a normal
        # box, so a short report whose title heads every page keeps its title once as search context (coordinator,
        # 4 Nov). Only the repeats go. A page number alone (no letter) goes everywhere.
        for (_, key), occurrences in found.items():
            if any(ch.isalpha() for ch in key):
                drop.discard(min((page, i) for _, page, i in occurrences))
        # Never all of a page's text: a page with no table whose every box would go (the same one line on every page,
        # a page that is only its header) keeps them.
        for read in reads:
            if not read.tables and read.boxes and all((read.page_number, i) in drop for i in range(len(read.boxes))):
                drop.difference_update((read.page_number, i) for i in range(len(read.boxes)))
        return drop

    def _read_page(self, *, page: Any, page_number: int) -> _PageRead:
        height = float(getattr(page, "height", 0.0) or 0.0)
        # Urdu pages (30 Oct): the word-by-word path below reads right-to-left lines left-to-right, so every line came
        # out letter-reversed ("یداینب" for "بنیادی") and Urdu search on Urdu PDFs failed. pdfplumber's right-to-left
        # reading gives the text in the right order; it would reverse English, so it runs only on mostly-Urdu pages.
        if self._mostly_arabic_script(page):
            # Read right to left, the English and the digits on the page come out backwards ("seY" for "Yes", "31."
            # for "13."); each such run is turned around again (18 Nov, reversed_text.turn_reversed_runs).
            text = turn_reversed_runs((page.extract_text(char_dir="rtl", char_dir_render="ltr") or "").strip())
            segments = [PDFSegment(page_number=page_number, segment_type="text", text=text, y0=0.0, x0=0.0)] if text \
                else []
            return _PageRead(page_number=page_number, height=height, path="rtl", boxes=segments, tables=[])
        table_segments: list[PDFSegment] = []
        table_bboxes: list[tuple[float, float, float, float]] = []

        tables = page.find_tables() or []
        for table in tables:
            bbox_raw = getattr(table, "bbox", None)
            if not bbox_raw:
                continue
            x0, y0, x1, y1 = self._normalize_bbox(bbox_raw)
            table_text = self._table_to_text(table.extract() or [])
            if not table_text:
                continue
            table_bboxes.append((x0, y0, x1, y1))
            table_segments.append(
                PDFSegment(
                    page_number=page_number,
                    segment_type="table",
                    text=table_text,
                    y0=y0,
                    x0=x0,
                )
            )

        boxes = self._layout_text_segments(page=page, page_number=page_number, table_bboxes=table_bboxes)
        # The layout path only when it is safe: its boxes must hold the page's text (review 3 Nov: a body inside a
        # Form XObject had no text box, so only the page number was kept) and the page must have no table without
        # ruling lines (read column by column, its names lost their numbers). Otherwise the word path, as before.
        # Widely tracked type comes out of the layout path one letter at a time ("e n s u r i n g"): word path too.
        letter_spaced = any(_SPACED_LETTERS.search(seg.text) for seg in boxes)
        if (boxes and not letter_spaced and self._layout_covers_page(page, table_bboxes)
                and not self._side_by_side_cells(page, table_bboxes)):
            return _PageRead(page_number=page_number, height=height, path="layout", boxes=boxes,
                             tables=table_segments)

        words = page.extract_words() or []
        filtered_words = [
            word
            for word in words
            if not self._word_in_any_table(word=word, table_bboxes=table_bboxes)
        ]
        text_segments = self._words_to_text_segments(
            words=filtered_words,
            page_number=page_number,
        )
        return _PageRead(page_number=page_number, height=height, path="words", boxes=text_segments,
                         tables=table_segments)

    @staticmethod
    def _layout_text_segments(
        *, page: Any, page_number: int, table_bboxes: list[tuple[float, float, float, float]]
    ) -> list[PDFSegment]:
        """One segment per pdfminer text box, in pdfminer's reading order; boxes inside a table are left out."""
        objects = getattr(page, "objects", None)
        boxes = objects.get("textboxhorizontal", []) if isinstance(objects, dict) else []
        size_of = PDFPlumberExtractor._font_sizer(page)
        segments: list[PDFSegment] = []
        for box in boxes:
            lines = [line.strip() for line in str(box.get("text", "")).splitlines() if line.strip()]
            # Rotated text (e.g. the World Bank's "Public Disclosure Authorized" margin stamp) comes out one letter a
            # line; joined it read "d e z i r o h t u A" and spoiled a Summarize (3 Nov). Left out, as the old path did.
            # Single LETTERS only: a column of short numbers ("45", "7") or markers ("a)") is kept (review 3 Nov).
            if not lines or (len(lines) >= 3 and sum(len(line) == 1 and line.isalpha() for line in lines)
                             > len(lines) / 2):
                continue
            # "ﬁ" as the letters "fi", as the word path writes it (pdfplumber's expand_ligatures): "efficient" matches.
            text = " ".join(lines).translate(_LIGATURES)
            x0, top, x1, bottom = (float(box.get(k, 0.0)) for k in ("x0", "top", "x1", "bottom"))
            cx, cy = (x0 + x1) / 2, (top + bottom) / 2
            if any(tx0 <= cx <= tx1 and ty0 <= cy <= ty1 for tx0, ty0, tx1, ty1 in table_bboxes):
                continue
            size = size_of(x0, top, x1, bottom)
            segments.append(PDFSegment(page_number=page_number, segment_type="text", text=text, y0=top, x0=x0,
                                       x1=x1, y1=bottom, size=size, end_size=size))
        return segments

    @staticmethod
    def _font_sizer(page: Any):
        """A function (x0, top, x1, bottom) -> median font size of the page's visible characters inside that box,
        0.0 when there are none. Characters are sorted by height once, so each box looks only at its own rows."""
        chars = []
        for c in getattr(page, "chars", None) or []:
            if not str(c.get("text", "")).strip():
                continue
            try:
                chars.append(((float(c["top"]) + float(c["bottom"])) / 2, (float(c["x0"]) + float(c["x1"])) / 2,
                              float(c["size"])))
            except (KeyError, TypeError, ValueError):
                continue
        chars.sort()
        mids = [c[0] for c in chars]

        def size_of(x0: float, top: float, x1: float, bottom: float) -> float:
            lo, hi = bisect.bisect_left(mids, top - 0.5), bisect.bisect_right(mids, bottom + 0.5)
            sizes = sorted(s for _, cx, s in chars[lo:hi] if x0 - 0.5 <= cx <= x1 + 0.5)
            return round(sizes[len(sizes) // 2], 2) if sizes else 0.0

        return size_of

    @staticmethod
    def _box_kind(segment: PDFSegment) -> str:
        """'table': a table segment (a wall: a run of labels and the search for its text stop there).
        'label': a letter, no digit, at most LABEL_MAX_WORDS words ("Sukkur", "Mirpur Khas", "vii) Universities").
        'value': no letter ("2019", "40", "171"), or a short label with its value ("Punjab 49%", "Figure 3"); left as it
        is (passage_filter drops what is too thin). 'context': anything longer, the text a label can join."""
        if segment.segment_type != "text":
            return "table"
        text = segment.text
        has_letter = any(ch.isalpha() for ch in text)
        if len(text.split()) > LABEL_MAX_WORDS:
            return "context"
        if not has_letter or any(ch.isdigit() for ch in text):
            return "value"
        return "label"

    @staticmethod
    def _box_gap(a: PDFSegment, b: PDFSegment) -> float:
        """Distance between two boxes' nearest edges; 0 when they touch or one lies over the other."""
        dx = max(0.0, a.x0 - b.x1, b.x0 - a.x1)
        dy = max(0.0, a.y0 - b.y1, b.y0 - a.y1)
        return (dx * dx + dy * dy) ** 0.5

    @classmethod
    def _join_lone_labels(cls, boxes: list[PDFSegment]) -> list[PDFSegment]:
        """Join each lone label box to the text beside it on the same page, so it is not a passage of its own.

        4 Nov, on the live app: "surajjj" got one-word passages as its closest ("Sukkur" from ASER 2023 p.30-31
        three times, "thousand" from Economic Survey ch.10 p.4): a short question's embedding lands near a near-empty
        passage. Each pdfminer text box was its own piece, so a chart legend, a map name or a word that justified
        spacing set apart became a passage. Decided by structure only, no word lists:

        A run of labels next to each other in pdfminer's reading order (number-only boxes between them don't break the
        run) goes, as one piece, to the context box just before or just after the run in reading order, whichever is
        nearer on the page (gap between the run's outline and the box; 0 when the box holds it). Reading order keeps a
        chart's legend with its own figure title; the gap sends a heading to the paragraph under it and a word set
        apart by spacing back into its paragraph. Before -> added after that box's text; after -> put in front of it.
        A table is a wall: a run stops there and never joins text on the table's other side; a run with no context
        box to join keeps its labels together as one piece. Value boxes stay where they are, so "Punjab 49%" is
        still findable on its own. The joined text keeps its page, so citations and open-at-page still point right.
        The word path (pages the layout path can't read safely) runs the same join on its blocks of lines, which come
        top to bottom: there a running header ("Education") or a line's clipped tail ("rnment.") was a lone piece too.
        """
        kinds = [cls._box_kind(b) for b in boxes]
        runs: list[list[int]] = []
        current: list[int] = []
        for i, kind in enumerate(kinds):
            if kind == "label":
                current.append(i)
            elif kind in ("context", "table") and current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        if not runs:
            return boxes

        before: dict[int, list[int]] = {}  # context index -> label boxes to put in front of it
        after: dict[int, list[int]] = {}  # context index -> label boxes to add after it
        own: dict[int, list[int]] = {}  # first label index of a run with nowhere to go -> the run's boxes
        joined: set[int] = set()
        for run in runs:
            outline = cls._outline([boxes[i] for i in run], text="")
            prev = next((i for i in range(run[0] - 1, -1, -1) if kinds[i] in ("context", "table")), None)
            nxt = next((i for i in range(run[-1] + 1, len(boxes)) if kinds[i] in ("context", "table")), None)
            prev = None if prev is None or kinds[prev] == "table" else prev  # never across a table
            nxt = None if nxt is None or kinds[nxt] == "table" else nxt
            joined.update(run)
            if prev is None and nxt is None:
                own[run[0]] = run
            elif nxt is None or (prev is not None and cls._box_gap(outline, boxes[prev])
                                 <= cls._box_gap(outline, boxes[nxt])):
                after.setdefault(prev, []).extend(run)
            else:
                before.setdefault(nxt, []).extend(run)

        out: list[PDFSegment] = []
        for i, box in enumerate(boxes):
            if i in own:
                out.append(cls._outline([boxes[j] for j in own[i]], text="\n".join(boxes[j].text for j in own[i])))
            elif i in before or i in after:
                members = [boxes[j] for j in (*before.get(i, []), i, *after.get(i, []))]
                out.append(cls._outline(members, text="\n".join(m.text for m in members)))
            elif i not in joined:
                out.append(box)
        return out

    @staticmethod
    def _next_to(a: PDFSegment, b: PDFSegment, *, after_heading: bool) -> str | None:
        """How box b, next in reading order, continues the text that ends with box a: " " (the same paragraph's next
        line), "\n" (the next paragraph, or the top of the next column), or None (a boundary: another page, a table,
        a heading, smaller type such as a caption or footnote, or text somewhere else on the page)."""
        if a.page_number != b.page_number or a.segment_type != "text" or b.segment_type != "text":
            return None
        if a.end_size and b.size:
            if b.size > a.end_size * SIZE_JUMP:
                return None  # bigger type: a heading starts here
            if b.size * SIZE_JUMP < a.end_size and not after_heading:
                return None  # smaller type after body text: a caption, a footnote, a chart's labels
        ref = max(a.end_size, b.size) or FALLBACK_SIZE
        gap = b.y0 - a.y1
        if b.x0 < a.x1 and b.x1 > a.x0 and gap >= -0.5 * ref:  # below a, in the same column
            if gap <= LINE_GAP * ref:
                return " "
            return "\n" if gap <= SECTION_GAP * ref else None
        if b.x0 >= a.x1 - 0.5 * ref and b.y0 <= a.y1:  # to the right, starting no lower: the next column
            return "\n"
        return None

    @classmethod
    def _join_paragraphs(cls, segments: list[PDFSegment], *, max_chars: int = PARAGRAPH_MAX_CHARS
                         ) -> list[PDFSegment]:
        """Join consecutive text boxes of a page, in reading order, into paragraph-sized pieces (up to max_chars).

        4 Nov demo rehearsal: a made-up 2-page policy PDF became 30 passages: one was its title alone, and the fee
        passage stopped before "by the 10th of each month". Its lines are spaced wide enough that pdfminer makes a
        text box of each line, and every box was chunked on its own, so a sentence split over boxes was split over
        passages and short boxes became passages of their own. Decided by structure only, no word lists:

        A box joins the piece before it when _next_to says it continues it: below it in the same column within
        SECTION_GAP font sizes, or at the top of the next column; and in the same size of type. A bigger box starts a
        new piece (a heading), and a heading takes the text under it even in smaller type; smaller type after body
        text (a caption, a footnote) starts a new piece. A new paragraph joins only while the piece has room for it
        (max_chars); the lines of one paragraph always stay together (the chunker windows a long piece, with overlap).
        A table is a wall and stays a piece of its own; pieces never cross pages, so each keeps its page number.
        The next line of a paragraph is joined with a space, the next paragraph with a line break."""
        out: list[PDFSegment] = []
        piece: list[PDFSegment] = []
        text = ""

        def close() -> None:
            if piece:
                out.append(piece[0] if len(piece) == 1 else cls._outline(piece, text=text))

        for seg in segments:
            if piece:
                # A heading so far: short, and every box in it in bigger type than this one.
                after_heading = bool(seg.size) and len(text) <= HEADING_MAX_CHARS and all(
                    m.size > seg.size * SIZE_JUMP and m.end_size > seg.size * SIZE_JUMP for m in piece)
                how = cls._next_to(piece[-1], seg, after_heading=after_heading)
                if how == "\n" and not after_heading and len(text) + 1 + len(seg.text) > max_chars:
                    how = None  # no room for another paragraph
                if how == " " and len(text) + 1 + len(seg.text) > PARAGRAPH_HARD_CAP * max_chars:
                    how = None  # spacing never changes on the page: don't let one piece swallow it all
                if how is not None:
                    piece.append(seg)
                    text = f"{text}{how}{seg.text}"
                    continue
                close()
            piece, text = [seg], seg.text
        close()
        return out

    @staticmethod
    def _outline(members: list[PDFSegment], *, text: str) -> PDFSegment:
        """One segment holding `text` over the outline of `members` (in reading order, all on one page): it starts
        with the first member's font size and ends with the last one's."""
        first = members[0]
        return PDFSegment(page_number=first.page_number, segment_type=first.segment_type, text=text,
                          y0=min(m.y0 for m in members), x0=min(m.x0 for m in members),
                          x1=max(m.x1 for m in members), y1=max(m.y1 for m in members),
                          size=first.size, end_size=members[-1].end_size)

    @staticmethod
    def _in_any(x: float, y: float, bboxes: list[tuple[float, float, float, float]]) -> bool:
        return any(bx0 <= x <= bx1 and by0 <= y <= by1 for bx0, by0, bx1, by1 in bboxes)

    @classmethod
    def _layout_covers_page(cls, page: Any, table_bboxes: list[tuple[float, float, float, float]]) -> bool:
        """The text boxes hold (almost) every character the page has outside its tables. A safety net: without
        all_texts pdfminer made no text box for text inside a Form XObject (pdfjam pages, stamping tools), which the
        word path does read; anything else the boxes miss falls back to the word path the same way."""
        def visible(t: Any) -> int:
            return sum(not ch.isspace() for ch in str(t or ""))

        on_page = sum(visible(c.get("text")) for c in (getattr(page, "chars", None) or [])
                      if not cls._in_any((float(c.get("x0", 0)) + float(c.get("x1", 0))) / 2,
                                         (float(c.get("top", 0)) + float(c.get("bottom", 0))) / 2, table_bboxes))
        objects = getattr(page, "objects", None)
        in_boxes = sum(visible(b.get("text")) for b in (objects.get("textboxhorizontal", []) if isinstance(
            objects, dict) else []) if not cls._in_any((float(b.get("x0", 0)) + float(b.get("x1", 0))) / 2,
                                                       (float(b.get("top", 0)) + float(b.get("bottom", 0))) / 2,
                                                       table_bboxes))
        return in_boxes >= 0.98 * on_page

    @classmethod
    def _side_by_side_cells(cls, page: Any, table_bboxes: list[tuple[float, float, float, float]] = ()) -> bool:
        """A table without ruling lines (find_tables misses it): a text box of short cells (3 words or fewer a line,
        2+ lines) whose lines sit on the same rows as short lines (3 words or fewer) beside it. pdfminer reads it
        column by column ("District Lahore Multan", then "Schools 45 12"); the word path keeps each row together
        ("Lahore 45"). Text inside a ruled table find_tables already read (table_bboxes) is left out of this check:
        before 20 Nov any page with a ruled table of short cells went to the word path, which reads two columns of
        text row by row. What this cannot tell apart: a borderless table whose cells beside are long text (read as
        columns now), two columns of short phrases on the same rows (still read as a table, row by row)."""
        objects = getattr(page, "objects", None)
        if not isinstance(objects, dict):
            return False

        def outside(o: Any) -> bool:  # 20 Nov: the cells of a ruled table (find_tables read it) are not text beside
            return not cls._in_any((float(o.get("x0", 0)) + float(o.get("x1", 0))) / 2,
                                   (float(o.get("top", 0)) + float(o.get("bottom", 0))) / 2, list(table_bboxes))

        lines = [line for line in objects.get("textlinehorizontal", []) or [] if outside(line)]
        for box in objects.get("textboxhorizontal", []) or []:
            if not outside(box):
                continue
            bx0, bx1, btop, bbot = (float(box.get(k, 0.0)) for k in ("x0", "x1", "top", "bottom"))
            own, beside = [], []
            for line in lines:
                x0, x1, top = float(line.get("x0", 0.0)), float(line.get("x1", 0.0)), float(line.get("top", 0.0))
                if bx0 <= (x0 + x1) / 2 <= bx1 and btop <= top + 0.5 and top <= bbot:
                    own.append(line)
                elif (x1 <= bx0 or x0 >= bx1) and len(str(line.get("text", "")).split()) <= 3:
                    # 20 Nov: only short cells beside count. A column of short lines next to a column of prose
                    # ("School is closed" | "The bus fee is not / charged for June and July.") is two columns of
                    # text; read row by row it came out "School is closed The bus fee is not".
                    beside.append(top)
            if len(own) < 2 or any(len(str(line.get("text", "")).split()) > 3 for line in own):
                continue
            aligned = sum(any(abs(float(line.get("top", 0.0)) - t) <= 2.0 for t in beside) for line in own)
            if aligned >= 2 and aligned > len(own) / 2:
                return True
        return False

    @staticmethod
    def _insert_tables(text_segments: list[PDFSegment], table_segments: list[PDFSegment]) -> list[PDFSegment]:
        """Text keeps its reading order; each table goes before the first text box that starts below it."""
        tables = sorted(table_segments, key=lambda t: (t.y0, t.x0))
        out: list[PDFSegment] = []
        for seg in text_segments:
            while tables and tables[0].y0 <= seg.y0:
                out.append(tables.pop(0))
            out.append(seg)
        return out + tables

    @staticmethod
    def _normalize_bbox(bbox: Any) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = bbox
        return float(x0), float(y0), float(x1), float(y1)

    @staticmethod
    def _table_to_text(rows: list[list[Any]]) -> str:
        normalized_rows: list[list[str]] = []
        for row in rows:
            if row is None:
                continue
            # A cell's text wrapped over lines ("Tuition\n(per month)") is one line: a line break inside a cell broke
            # the markdown row in two (20 Nov).
            values = [" ".join(str(cell).split()) if cell is not None else "" for cell in row]
            if any(values):
                normalized_rows.append(values)
        if not normalized_rows:
            return ""

        max_cols = max(len(row) for row in normalized_rows)
        padded = [row + [""] * (max_cols - len(row)) for row in normalized_rows]
        # 20 Nov: the table's own first row is its header when it looks like one (every cell filled, no cell a bare
        # number or an amount, more rows under it), so each part of a long table carries the real column names
        # ("Monthly bus fee"), not col_1, col_2 (passage_cleanup.keep_tables_whole repeats the header row in every
        # part).
        names = [f"col_{idx + 1}" for idx in range(max_cols)]
        if len(padded) > 1 and all(padded[0]) and not any(
                _BARE_NUMBER.fullmatch(c) or _AMOUNT.search(c) for c in padded[0]):
            names, padded = padded[0], padded[1:]
        if pd is not None:
            frame = pd.DataFrame(padded, columns=names)
            try:
                markdown = frame.to_markdown(index=False)
                return f"Table:\n{markdown}".strip()
            except Exception:
                # pandas.to_markdown may require optional `tabulate`; fallback to manual markdown.
                pass

        header = " | ".join(names)
        separator = " | ".join("---" for _ in range(max_cols))
        row_lines = "\n".join(f"| {' | '.join(row)} |" for row in padded)
        return f"Table:\n| {header} |\n| {separator} |\n{row_lines}".strip()

    @staticmethod
    def _word_in_any_table(
        *,
        word: dict[str, Any],
        table_bboxes: list[tuple[float, float, float, float]],
    ) -> bool:
        if not table_bboxes:
            return False
        x0 = float(word.get("x0", 0.0))
        x1 = float(word.get("x1", x0))
        y0 = float(word.get("top", 0.0))
        y1 = float(word.get("bottom", y0))
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        for tx0, ty0, tx1, ty1 in table_bboxes:
            if tx0 <= cx <= tx1 and ty0 <= cy <= ty1:
                return True
        return False

    @staticmethod
    def _words_to_text_segments(
        *,
        words: list[dict[str, Any]],
        page_number: int,
    ) -> list[PDFSegment]:
        if not words:
            return []

        ordered_words = sorted(
            words,
            key=lambda word: (float(word.get("top", 0.0)), float(word.get("x0", 0.0))),
        )

        lines: list[dict[str, Any]] = []
        current_words: list[dict[str, Any]] = []
        current_top: float | None = None
        line_tolerance = 2.5

        for word in ordered_words:
            top = float(word.get("top", 0.0))
            if current_top is None or abs(top - current_top) <= line_tolerance:
                current_words.append(word)
                current_top = top if current_top is None else min(current_top, top)
                continue

            line = PDFPlumberExtractor._finalize_line(current_words=current_words)
            if line:
                lines.append(line)
            current_words = [word]
            current_top = top

        line = PDFPlumberExtractor._finalize_line(current_words=current_words)
        if line:
            lines.append(line)
        if not lines:
            return []

        segments: list[PDFSegment] = []
        block_lines: list[dict[str, Any]] = []
        max_line_gap = 8.0

        for line in lines:
            if not block_lines:
                block_lines.append(line)
                continue

            previous = block_lines[-1]
            if line["y0"] - previous["y1"] <= max_line_gap:
                block_lines.append(line)
                continue

            segments.append(
                PDFPlumberExtractor._lines_to_segment(
                    lines=block_lines,
                    page_number=page_number,
                )
            )
            block_lines = [line]

        if block_lines:
            segments.append(
                PDFPlumberExtractor._lines_to_segment(
                    lines=block_lines,
                    page_number=page_number,
                )
            )

        return [segment for segment in segments if segment.text.strip()]

    @staticmethod
    def _finalize_line(*, current_words: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not current_words:
            return None
        ordered = sorted(current_words, key=lambda word: float(word.get("x0", 0.0)))
        text = " ".join(str(word.get("text", "")).strip() for word in ordered).strip()
        if not text:
            return None
        return {
            "text": text,
            "x0": min(float(word.get("x0", 0.0)) for word in ordered),
            "y0": min(float(word.get("top", 0.0)) for word in ordered),
            "x1": max(float(word.get("x1", word.get("x0", 0.0))) for word in ordered),
            "y1": max(float(word.get("bottom", 0.0)) for word in ordered),
        }

    @staticmethod
    def _lines_to_segment(*, lines: list[dict[str, Any]], page_number: int) -> PDFSegment:
        text = "\n".join(line["text"] for line in lines).strip()
        return PDFSegment(
            page_number=page_number,
            segment_type="text",
            text=text,
            y0=float(lines[0]["y0"]),
            x0=float(min(line["x0"] for line in lines)),
            x1=float(max(line.get("x1", line["x0"]) for line in lines)),
            y1=float(max(line["y1"] for line in lines)),
        )

    def _dedupe_adjacent_segments(
        self, segments: list[PDFSegment], *, keep_text_repeats: bool = False
    ) -> list[PDFSegment]:
        if not segments:
            return []

        deduped: list[PDFSegment] = [segments[0]]
        for segment in segments[1:]:
            previous = deduped[-1]
            both_text = previous.segment_type == segment.segment_type == "text"
            if previous.page_number == segment.page_number and not (keep_text_repeats and both_text):
                score = self._similarity_ratio(previous.text, segment.text)
                if score >= self._dedupe_threshold:
                    continue
            deduped.append(segment)
        return deduped

    @staticmethod
    def _similarity_ratio(left: str, right: str) -> float:
        if _rapidfuzz is not None:
            return float(_rapidfuzz.ratio(left, right))
        return difflib.SequenceMatcher(None, left, right).ratio() * 100.0

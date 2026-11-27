"""Clean a document's passages before they are embedded: no tiny pieces, no repeats, no page furniture, no YAML.

18 Nov, read-only audit of the live index (9,342 passages, 80 documents): 36% were junk, ~99% of it from PDFs.
  - tiny pieces under 40 characters (34%): a line's last word ("problems?", ASER 2023 p.169), a label, a heading on
    its own. The PDF reader gives a page as several pieces (text boxes, paragraphs, tables) and each piece was split on
    its own, so a short piece became a short passage.
  - exact repeats inside one document (10.8%): mostly page headers ("Pakistan Economic Survey 2025-26", "Education").
  - YAML front matter of Markdown files as a passage of its own ("---\\nname: ...\\ndescription: ...").
A tiny passage sits close to any short question and pushes real answers out (4 Nov: "okay" matched "o", "not.").

What happens here, in order (tidy_segments, then tidy_passages after the splitter):
  1. Front matter at the start of a document is taken out; its title/name and description stay as one plain line in
     front of the body, so the file is still found by what it says it is (strip_front_matter).
  2. Page furniture: a short line at the top or bottom of a PDF page that repeats there on many pages (a running
     header or footer, a page number) is removed; the first page that has it keeps it (drop_repeated_page_lines). The
     PDF reader already does this by position on the page (pdf_extractor._running_headers); this is the same rule on
     the text, for pages its position test can't see (word-path pages, headers inside a table's area).
  3. Thin passages ("o", "4", "Ibid.") are dropped (passage_filter.is_thin, 4 Nov).
  4. Exact repeats in the same document (same words, any spacing or case) are stored once.
  5. A passage under MIN_PASSAGE_CHARS joins a neighbour on its page: a tail (it ends a sentence or starts in lower
     case) joins the one before, anything else (a heading, a label) the one after. A tiny piece (under TINY_CHARS, or
     under TINY_WORDS words) alone on its page that starts in lower case is the end of the page before ("e energy
     sources."): it joins the passage before it. Any other piece alone on its page ("Sindh 49%", "Annex 3") stays a
     passage of its own with its own page number, as decided on 4 Nov (passage_filter.py): a page is never cited
     for text that is not on it.
Then passages are numbered chunk-0, chunk-1, ... in order, as before.

What this cannot catch: a header that differs on every page, near-duplicates (one number different), junk longer
than MIN_PASSAGE_CHARS (a row of chart labels). Reversed text is handled when the PDF is read (extract.py).
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

from src.rag.passage_filter import is_thin

MIN_PASSAGE_CHARS = 100  # shorter passages join a neighbour on their page
TINY_CHARS = 40  # the audit's "tiny": under 40 characters, or under 5 words; joins a neighbour even across a page
TINY_WORDS = 5
TABLE_TITLE_CHARS = 120  # a heading or caption right before a table: kept with every part of the table
TABLE_TITLE_LINES = 3

# Running lines (step 2): the same thresholds as the PDF reader's position rule (pdf_extractor HEADER_MIN_*).
LINE_MAX_CHARS = 100
EDGE_LINES = 2  # lines looked at on each edge of a page
LINE_MIN_SHARE = 0.3
LINE_MIN_PAGES = 3

_EDGE_DIGITS = re.compile(r"^\d+|\d+$")
_PAGE_WORD = re.compile(r"\bpage\s+\d+")
_FRONT_MATTER = re.compile(r"\A﻿?\s*---[ \t]*\n(.*?)\n(?:---|\.\.\.)[ \t]*(?:\n|\Z)", re.DOTALL)
_KEY = re.compile(r"^([A-Za-z_][\w-]*)[ \t]*:[ \t]*(.*)$")
_TAIL_END = re.compile(r"[.!?;,)\]\"'’”%]\s*$")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_DIVIDER = re.compile(r"^\s*\|(\s*:?-{3,}:?\s*\|)+\s*$")


# --- 1. front matter ------------------------------------------------------------------------------------------------

def strip_front_matter(text: str) -> str:
    """The text without a YAML front matter block at its very start. Its title (or name) and description stay as one
    plain line in front of the body; every other key (tags, allowed tools, dates) goes."""
    m = _FRONT_MATTER.match(text or "")
    if not m:
        return text
    lines = m.group(1).split("\n")
    first = next((ln for ln in lines if ln.strip()), "")
    if not _KEY.match(first):  # "---" then prose: a horizontal rule, not front matter
        return text
    values: dict[str, str] = {}
    key = None
    for line in lines:
        km = _KEY.match(line)
        if km and not line[:1].isspace():
            key = km.group(1).lower()
            values[key] = km.group(2).strip()
        elif key is not None and line.strip():  # an indented or folded continuation of the last key
            values[key] = f"{values[key]} {line.strip()}".strip()
    for k, v in values.items():
        v = v.lstrip(">|-+ ").strip()
        values[k] = v[1:-1] if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'" else v
    title = values.get("title") or values.get("name") or ""
    lead = ": ".join(p for p in (title, values.get("description", "")) if p)
    body = text[m.end():].lstrip("\n")
    return f"{lead}\n\n{body}" if lead else body


# --- 2. running headers and footers ---------------------------------------------------------------------------------

def _line_key(line: str) -> str:
    """A line with spacing collapsed, case folded, and a page number in it as "#": a run of digits at its start or
    end, or after "page" ("Education | 87" and "Education | 88" are the same footer; a page number alone is "#").
    Digits elsewhere stay, so "On page 3, 21 schools reopened" and "..., 28 schools reopened" are different lines."""
    key = " ".join(line.split()).casefold()
    key = _PAGE_WORD.sub("page #", key)
    return _EDGE_DIGITS.sub("#", key)


def drop_repeated_page_lines(segments: list[tuple[int | None, str]]) -> list[tuple[int | None, str]]:
    """Segments without the short lines repeated at the top or bottom of many pages (see the module notes).

    Only for documents with page numbers on at least LINE_MIN_PAGES pages. A line is page furniture when its key is
    among the first or last EDGE_LINES lines of at least max(LINE_MIN_PAGES, LINE_MIN_SHARE of the pages) pages. It
    goes from those edges only, never from the middle of a page; the earliest page keeps it when it has a letter (a
    report's title heading every page stays once), a page number alone goes everywhere. A page never loses all its
    text this way."""
    pages: dict[int, list[int]] = defaultdict(list)  # page -> indexes of its segments, in order
    for i, (page, text) in enumerate(segments):
        if page is not None and (text or "").strip():
            pages[page].append(i)
    if len(pages) < LINE_MIN_PAGES:
        return segments
    need = max(LINE_MIN_PAGES, math.ceil(LINE_MIN_SHARE * len(pages)))

    lines = {i: segments[i][1].split("\n") for idx in pages.values() for i in idx}
    edges: dict[int, list[tuple[int, int]]] = {}  # page -> [(segment index, line index)] of its edge lines
    for page, idx in pages.items():
        flat = [(i, j) for i in idx for j, ln in enumerate(lines[i]) if ln.strip()]
        edge = flat[:EDGE_LINES] + [p for p in flat[-EDGE_LINES:] if p not in flat[:EDGE_LINES]]
        edges[page] = [(i, j) for i, j in edge if len(lines[i][j].strip()) <= LINE_MAX_CHARS]

    seen: dict[str, set[int]] = defaultdict(set)
    for page, edge in edges.items():
        for i, j in edge:
            seen[_line_key(lines[i][j])].add(page)
    furniture = {k for k, p in seen.items() if len(p) >= need}
    if not furniture:
        return segments

    drop: set[tuple[int, int]] = set()
    for page in sorted(pages):
        hits = [(i, j) for i, j in edges[page] if _line_key(lines[i][j]) in furniture]
        hits = [(i, j) for i, j in hits
                if not (any(ch.isalpha() for ch in _line_key(lines[i][j]))
                        and page == min(seen[_line_key(lines[i][j])]))]
        total = sum(1 for i in pages[page] for ln in lines[i] if ln.strip())
        if hits and len(hits) < total:  # never all of a page's text
            drop.update(hits)
    if not drop:
        return segments

    out = []
    for i, (page, text) in enumerate(segments):
        if i in lines and any((i, j) in drop for j in range(len(lines[i]))):
            text = "\n".join(ln for j, ln in enumerate(lines[i]) if (i, j) not in drop).strip()
            if not text:
                continue
        out.append((page, text))
    return out


def tidy_segments(segments: list[tuple[int | None, str]]) -> list[tuple[int | None, str]]:
    """Steps 1 and 2: front matter off the document's first piece, running headers and footers off its pages."""
    segments = [(p, t) for p, t in segments if t and t.strip()]
    if segments:
        page, first = segments[0]
        segments[0] = (page, strip_front_matter(first))
    return drop_repeated_page_lines(segments)


# --- 2b. tables keep their title and header row ---------------------------------------------------------------------

def _is_table_row(line: str) -> bool:
    return bool(_TABLE_ROW.match(line))


def _table_at(lines: list[str], i: int) -> int:
    """The end (exclusive) of a markdown table whose header row is lines[i] (a row, then a divider, then rows), or i
    when there is none there."""
    if i + 1 >= len(lines) or not _is_table_row(lines[i]) or not _TABLE_DIVIDER.match(lines[i + 1]):
        return i
    end = i + 2
    while end < len(lines) and _is_table_row(lines[end]) and not _TABLE_DIVIDER.match(lines[end]):
        end += 1
    return end


def _title_lines(before: list[str]) -> int:
    """How many of the last lines before a table are its title: the block of lines just above it (a heading or
    caption, the "Table:" label; blank lines between it and the table are skipped, a blank line above it ends it), when
    that block is at most TABLE_TITLE_LINES lines and TABLE_TITLE_CHARS characters; 0 when it is longer (a
    paragraph is not a title)."""
    n = len(before)
    while n and not before[n - 1].strip():
        n -= 1
    top = n
    while top and before[top - 1].strip():
        top -= 1
    block = before[top:n]
    if not block or len(block) > TABLE_TITLE_LINES or sum(len(ln.strip()) for ln in block) > TABLE_TITLE_CHARS:
        return 0
    return len(before) - top


def _table_parts(text: str, *, chunk_size: int) -> list[str]:
    """text as one piece, or, when it holds a table too long for one passage, as pieces: the text before the table,
    the table in parts of whole rows that each start with the table's title and header row, the text after it."""
    if len(text) <= chunk_size:
        return [text]
    lines = text.split("\n")
    out: list[str] = []
    prose: list[str] = []
    cut = False
    i = 0
    while i < len(lines):
        end = _table_at(lines, i)
        if end == i:
            prose.append(lines[i])
            i += 1
            continue
        n = _title_lines(prose)
        title = [ln for ln in prose[len(prose) - n:] if ln.strip()] if n else []
        head = [*title, lines[i], lines[i + 1]]
        rows = lines[i + 2:end]
        if len("\n".join(head + rows)) <= chunk_size:  # the table fits in one passage: left as it is
            prose.extend(lines[i:end])
            i = end
            continue
        before = "\n".join(prose[:len(prose) - n]).strip()
        if before:
            out.append(before)
        prose, cut = [], True
        part: list[str] = []
        for row in rows:
            if part and len("\n".join(head + part + [row])) > chunk_size:
                out.append("\n".join(head + part))
                part = []
            part.append(row)
        if part:
            out.append("\n".join(head + part))
        i = end
    if not cut:
        return [text]
    rest = "\n".join(prose).strip()
    if rest:
        out.append(rest)
    return out


def split_table_head(text: str) -> tuple[str, str] | None:
    """(head, rest) when a passage starts with a table's head as keep_tables_whole copies it into every part of a long
    table: up to TABLE_TITLE_LINES + 1 short title lines (a heading, the "Table:" label; no row; blank lines between
    them and before the header row are allowed: "By paragraphs and sentences" keeps the blank line after a heading,
    review round 2), the header row and the divider, then at least one row. None otherwise.

    Review 20 Nov: parts 2+ of a long table start with that copied head, which never sits next to their rows in the
    document, so "Open document" could not find them (originals.locate looks for the rows alone then) and text rebuilt
    from passages showed the head once per part (originals.join_passages drops the repeat)."""
    lines = (text or "").strip().split("\n")
    title = 0  # non-blank title lines so far
    for k in range(len(lines) - 2):
        if _table_at(lines, k) > k + 2:
            return "\n".join(lines[:k + 2]), "\n".join(lines[k + 2:])
        line = lines[k].strip()
        if not line:
            continue
        title += 1
        if (_is_table_row(line) or title > TABLE_TITLE_LINES + 1
                or sum(len(ln.strip()) for ln in lines[:k + 1]) > TABLE_TITLE_CHARS):
            return None
    return None


def _is_title_piece(text: str) -> bool:
    t = text.strip()
    return bool(t) and len(t) <= TABLE_TITLE_CHARS and len(t.split("\n")) <= TABLE_TITLE_LINES and not any(
        _is_table_row(line) for line in t.split("\n"))


def keep_tables_whole(segments: list[tuple[int | None, str]], *, chunk_size: int) -> list[tuple[int | None, str]]:
    """Step 2b: a table never loses its title or its header row to the splitter.

    20 Nov (issues.json, "Greenfield PDF still answers from 30 small passages"): a fee table longer than one passage
    was cut by the splitter in the middle of a row; the rows after the cut had no heading ("School bus fee") and no
    column names ("Monthly bus fee"), so "what is the monthly bus fee" never found them. Decided by structure only:
      - a short piece (a heading or caption: TABLE_TITLE_CHARS, TABLE_TITLE_LINES) right before a table piece on the
        same page goes in front of the table (the PDF reader gives a table as a piece of its own, "Table:" first);
      - a table that doesn't fit in one passage (chunk_size) is cut between rows, never inside one, and every part
        starts with the table's title and its header row. A table that fits, and all other text, is left as it is.
    What this cannot catch: a table without a header row and divider (a borderless table the PDF reader reads as
    lines), a title further up the page than the piece right before the table, a single row longer than a passage."""
    out: list[tuple[int | None, str]] = []
    for page, text in segments:
        if (out and out[-1][0] == page and text.lstrip().startswith("Table:")
                and _is_title_piece(out[-1][1])):
            text = f"{out.pop()[1].strip()}\n{text.lstrip()}"
        out.extend((page, part) for part in _table_parts(text, chunk_size=chunk_size))
    return out


# --- 3-5. the passages ----------------------------------------------------------------------------------------------

def _key(text: str) -> str:
    return " ".join(text.split()).casefold()


def is_tiny(text: str) -> bool:
    t = (text or "").strip()
    return len(t) < TINY_CHARS or len(t.split()) < TINY_WORDS


def _is_tail(text: str) -> bool:
    """A piece that ends something (a sentence, a line's last words) rather than starts it (a heading, a label)."""
    t = text.strip()
    return bool(_TAIL_END.search(t)) or t[:1].islower()


def _continues(text: str) -> bool:
    """A piece that starts in lower case: the end of a line or word from the page before ("problems?")."""
    return text.strip()[:1].islower()


def _join(a: str, b: str) -> str:
    return f"{a.rstrip()}\n\n{b.lstrip()}"


def tidy_passages(chunks: list, *, chunk_size: int) -> list:
    """Steps 3-5 on a document's chunks (anything with .text and .page_number, in reading order); renumbers them.

    A short passage joins a neighbour only while the result stays within chunk_size + MIN_PASSAGE_CHARS (the
    embedding model, bge-m3, reads 8,192 tokens, far above that); a tiny one (is_tiny) joins a neighbour on its page
    even past it, so only a page whose whole text is tiny keeps a tiny passage."""
    cap = chunk_size + MIN_PASSAGE_CHARS
    short = min(MIN_PASSAGE_CHARS, chunk_size // 4)  # "short" next to the passage size (tests use tiny sizes)
    seen: set[str] = set()
    kept = []
    for c in chunks:
        if is_thin(c.text):
            continue
        k = _key(c.text)
        if k in seen:
            continue
        seen.add(k)
        kept.append(c)

    out: list = []
    carry = None  # a short passage waiting to go in front of the next one on its page

    def fits_prev(piece) -> bool:
        return bool(out) and out[-1].page_number == piece.page_number and len(out[-1].text) + len(piece.text) <= cap

    def place(piece) -> None:
        """A short passage that can't go in front of the next one: the one before on its page (a tiny one even past
        the cap), else (a tiny piece that continues a line, "e energy sources.") the one before on the page before,
        else a passage of its own."""
        same_page = bool(out) and out[-1].page_number == piece.page_number
        if fits_prev(piece) or (out and is_tiny(piece.text) and (same_page or _continues(piece.text))):
            out[-1].text = _join(out[-1].text, piece.text)
        else:
            out.append(piece)

    for c in kept:
        if carry is not None:
            same_page = c.page_number == carry.page_number
            # 20 Nov: a tiny piece (a heading, "6. Contact") goes in front of the next passage on its page even past
            # the cap: a passage under TINY_CHARS is never left alone while its page has other text.
            if same_page and (len(carry.text) + len(c.text) <= cap or is_tiny(carry.text)):
                c.text = _join(carry.text, c.text)
            else:
                place(carry)
            carry = None
        if len(c.text.strip()) < short:
            if fits_prev(c) and _is_tail(c.text):
                out[-1].text = _join(out[-1].text, c.text)
            else:
                carry = c
            continue
        out.append(c)
    if carry is not None:
        place(carry)

    final, keys = [], set()
    for c in out:  # a join can, rarely, make two passages the same
        k = _key(c.text)
        if k in keys:
            continue
        keys.add(k)
        c.chunk_id = f"chunk-{len(final)}"
        final.append(c)
    return final

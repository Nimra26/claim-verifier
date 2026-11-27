"""Passages with too little real text to answer anything ("thin"): kept out of the index and out of every search.

4 Nov, on the live app: "okay" got five junk passages as its closest, ASER p.166 "o" (70% similar), p.168
"not." (68%), World Bank Girls' Education p.1 "i" (65%) three times. The PDF reader's layout path
(pdf_extractor._layout_text_segments) keeps every pdfminer text box as its own piece, and RAGIngestionService chunks
each piece on its own, so a box of one letter became a passage of one letter: a bullet glyph set as the letter "o", the
tail of a wrapped line ("not."), the rotated "Public Disclosure Authorized" stamp one letter a box, page numbers, lone
chart labels. A short question's embedding sits close to a near-empty passage's, so they won.

Measured on the live database (4 Nov, read-only): 9,312 passages in 79 documents; 1,805 thin by this rule, all in 8
PDFs (Economic Survey ch.10: 505 of 824). By letters alone, 1,649 passages had no letter at all ("•", "4", "0%",
"2023"), and of those with 1-4 letters nearly all were one or two words ("KP", "Ibid.", "iii.", "to").

The rule: thin when it has fewer than MIN_LETTERS letters AND fewer than MIN_WORDS words, unless it holds both a
letter and a digit (a label with its value: "KP 49%"). So "Sindh 49%", "29% male", "Punjab", "Methodology", a row of
numbers ("| 2019-20 | 12038.8 | 23758.2 |", many words) and Urdu words all stay. Single headings like "Punjab" stay on
purpose: they are words a person may ask about. MIN_LETTERS 10 would also drop 245 of them ("Sindh", "Total").
Since 4 Nov ("surajjj" -> "Sukkur", "thousand") the PDF reader joins such a lone word box to the text beside it on its
page (pdf_extractor._join_lone_labels), so newly read PDFs rarely have one; documents indexed before keep theirs until
they are read again.

NOT_THIN_SQL is the same rule in Postgres, for the search (pgvector_store.similarity_search). Postgres' [[:alpha:]]
counts Urdu letters as Python does; it also counts Arabic-Indic digits (١٢) as letters, which only keeps a few more.
Both rules agree on all 9,312 live passages (checked 4 Nov). A passage of MAX_THIN_CHARS characters or more is never
thin (the longest thin one on the live database had 10): Postgres reads the length first and skips the regular
expressions for most rows. Without that the search went from 45 ms to 560 ms (EXPLAIN ANALYZE, live database, all
documents allowed, a sequential scan: the expressions ran on every row).
"""

from __future__ import annotations

import re

MIN_LETTERS = 5
MIN_WORDS = 3
MAX_THIN_CHARS = 40


def is_thin(text: str | None) -> bool:
    t = text or ""
    if len(t) >= MAX_THIN_CHARS:
        return False
    letters = sum(ch.isalpha() for ch in t)
    if letters >= MIN_LETTERS or len(t.split()) >= MIN_WORDS:
        return False
    return not (letters and any(ch.isdigit() for ch in t))


def drop_thin(chunks: list) -> list:
    """The chunks (anything with .text) that are not thin, in order."""
    return [c for c in chunks if not is_thin(c.text)]


# Postgres, on the rag_chunks column `text`; a SQL condition true for a passage worth searching. Written with plain
# colons for psql; in SQLAlchemy text() use NOT_THIN_SQL_BOUND, where each colon is escaped so "[[:alpha:]]" is not
# read as a bind parameter.
NOT_THIN_SQL = (  # the cheap length test first: OR stops at the first true one
    f"(length(text) >= {MAX_THIN_CHARS} "
    f"OR length(regexp_replace(text, '[^[:alpha:]]', '', 'g')) >= {MIN_LETTERS} "
    f"OR cardinality(regexp_split_to_array(btrim(text, E' \\t\\n\\r'), '\\s+')) >= {MIN_WORDS} "
    "OR (text ~ '[[:alpha:]]' AND text ~ '[[:digit:]]'))"
)
NOT_THIN_SQL_BOUND = NOT_THIN_SQL.replace(":", "\\:")


# --- Read time: what the model reads and the page lists (18 Nov) ---------------------------------------------------
# 18 Nov, owner on the live app, "snow kea ha?": the five closest passages were "problems?" (51%), "human" (48%),
# "slamabad." (48%) from ASER 2023 and Economic Survey ch.10, and a skill file's front matter (37%). ASER p.169's
# "problems?" is the tail of the line "...the child has any of the following health problems?": the PDF reader's
# layout path kept that line's clipped tail (its own pdfminer text box) as a piece, and the chunker (fixed_window)
# chunks each piece on its own. Since 4 Nov pdf_extractor._join_lone_labels joins such a box to its paragraph, but
# documents read before keep theirs until they are read again (likely here, both were added 30 Oct; the live database
# was not read to check). Issue: re-index ASER 2023 and Economic Survey ch.10.
# is_thin keeps one real word on purpose (a heading like "Punjab" stays searchable), but a passage that short never
# answers anything and matches any short question: it is never read by the model or listed under an answer.
MIN_READ_WORDS = 5
MIN_READ_CHARS = 40
_FRONT_FENCES = ("---", "+++")
_FRONT_KEY = re.compile(r"^[A-Za-z_][\w.-]*\s*:(\s|$)")
_FRONT_ITEM = re.compile(r"^-\s+\S")  # a YAML list under a key ("tags:" then "- a")


def too_short_to_read(text: str | None) -> bool:
    """Fewer than MIN_READ_WORDS words AND under MIN_READ_CHARS characters: "problems?", "human", "Sindh 49%".
    A row of figures ("| 2019-20 | 12038.8 | 23758.2 |") or a short sentence stays."""
    t = (text or "").strip()
    return len(t) < MIN_READ_CHARS and len(t.split()) < MIN_READ_WORDS


def metadata_only(text: str | None) -> bool:
    """A file's front matter and nothing else: "key: value" lines after a --- (or +++) fence, e.g. a skill file's
    "---", "name: x", "description: y", "---". It says what the file is, not what it holds. A passage with any line
    of prose after the fence is not metadata only."""
    lines = [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]
    if len(lines) < 2 or lines[0] not in _FRONT_FENCES:
        return False
    body = [ln for ln in lines[1:] if ln not in _FRONT_FENCES]
    return all(_FRONT_KEY.match(ln) or _FRONT_ITEM.match(ln) for ln in body)


def unreadable(text: str | None) -> bool:
    """Never sent to the model nor listed under an answer: too short to read, or front matter only."""
    return too_short_to_read(text) or metadata_only(text)

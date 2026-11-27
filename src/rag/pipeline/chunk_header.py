"""Contextual chunk headers: a chunk is EMBEDDED with its document's name (and page or slide) in front of it.

Why (3 Nov, public eval Q5): "How many children aged 5-16 are out of school according to the 2023 Census ...?" The
answer is one row of the Census table file, but the words "Census 2023" appear only in the file's name and first
chunk, so the row's chunk didn't say which document it came from and the ITA ASER PDF ranked first. The light form
of contextual retrieval (no model call): "Document: PBS Census 2023 Table12 Literacy Enrolment OOSC National · page 12".

Only the embedding input changes. The stored chunk text, citations, the proof view and the answer prompt are as
before, and so is the keyword index (rag_chunks.tsv is generated from the stored text). A question is embedded as
it is typed: embed_query never sees a header.

The header is deliberately NOT in the keyword index (3 Nov): there every chunk of a file would match the words of its
name equally (a question saying "census" or "2023" would pull up all of that file's chunks in no useful order and push
real word matches out), adding it needs a migration of the generated tsv column, and the Ask Claude guard's copied-
words check (intelligence/corpus.py PHRASE_SQL) reads the same tsv, so file names would start blocking questions.
Measure the embedding-only change on the eval first.

Each chunk records how it was embedded (meta.embed_input in rag_chunks: HEADER or BARE; no value = BARE, the
behaviour before 3 Nov), so src/modules/workspace/reembed.py re-embeds only the chunks made the other way.
"""

from __future__ import annotations

import os
import re

from src.rag.models import RAGChunk

HEADER = "header"  # embedded as chunk_header(...) + blank line + text
BARE = "text"  # embedded as the text alone (every chunk indexed before 3 Nov)
SLIDE_EXTENSIONS = frozenset({".pptx", ".ppt", ".odp", ".key"})  # their page numbers are slide numbers (extract.py)
# The names RAGIngestionService gives text that came without one: not a document title, so no header.
PLACEHOLDER_SOURCES = frozenset({"inline-text", "uploaded-pdf"})
_EXTENSION = re.compile(r"\.[A-Za-z0-9]{1,8}")


def _file_name(source: str | None) -> str:
    # "<group>/uploaded/<file>", "<group>/<file>", "my_<id>/uploaded/<file>": the file is the last part.
    return re.split(r"[\\/]", source or "")[-1].strip()


def _split(name: str) -> tuple[str, str]:
    stem, ext = os.path.splitext(name)
    if stem and _EXTENSION.fullmatch(ext):
        return stem, ext.lower()
    return name, ""  # "Report v1.2 final" keeps its ".2 final"


def document_title(source: str | None) -> str | None:
    """The file name as a title: no folder, no extension, _ and - as spaces. None when there is no real name."""
    name = _file_name(source)
    if not name or name in PLACEHOLDER_SOURCES:
        return None
    stem, _ = _split(name)
    title = " ".join(re.sub(r"[_\-]+", " ", stem).split())
    return title or None


def chunk_header(source: str | None, page_number: int | None = None) -> str | None:
    """'Document: <title>', plus ' · page N' (' · slide N' for a deck) when the chunk has a page. None: no header."""
    title = document_title(source)
    if title is None:
        return None
    header = f"Document: {title}"
    if page_number is not None:
        word = "slide" if _split(_file_name(source))[1] in SLIDE_EXTENSIONS else "page"
        header += f" · {word} {page_number}"
    return header


def embed_input(chunk: RAGChunk, *, headers: bool) -> str:
    """What the embedding model reads for this chunk. The chunk's own text is never changed."""
    header = chunk_header(chunk.source, chunk.page_number) if headers else None
    return f"{header}\n\n{chunk.text}" if header else chunk.text

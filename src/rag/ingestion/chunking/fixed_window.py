from __future__ import annotations

import re

from src.rag.ingestion.chunking.base import ChunkingStrategy
from src.rag.models import RAGChunk

# 18 Nov audit: the windows were cut at a fixed character count, so passages began and ended inside words
# ("e energy sources.", "ons of classifying viruses."). A window now ends at its last sentence end, else line break,
# else space, found in its last BOUNDARY_SHARE (so a window is never much shorter than chunk_size), and the next one
# starts at a word. At most chunk_overlap characters are repeated, as before, so originals.join_passages still rebuilds
# the text. Text without such a boundary (a long URL, a run of digits) is cut where it was before.
BOUNDARY_SHARE = 0.25
_SENTENCE_END = re.compile(r"[.!?:;]['\")\]]?(?=\s)")


def _cut(text: str, start: int, end: int) -> int:
    """Where the window [start, end) should end: at the whitespace after a sentence end, else at a line break, else at
    a space, the last one in the window's last BOUNDARY_SHARE; end itself when there is none."""
    if end >= len(text):
        return len(text)
    lo = max(start + 1, end - int((end - start) * BOUNDARY_SHARE))
    window = text[lo:end + 1]  # one past the end: whitespace right at end is a boundary too
    ends = [m.end() for m in _SENTENCE_END.finditer(window)]
    best = ends[-1] if ends else window.rfind("\n")
    if best < 0:
        best = max(window.rfind(" "), window.rfind("\t"))
    return lo + best if best > 0 else end


def _word_start(text: str, pos: int, limit: int) -> int:
    """pos, moved forward to the start of the next word when it falls inside a word or on spaces (never to limit)."""
    if pos <= 0 or pos >= len(text) or (text[pos - 1].isspace() and not text[pos].isspace()):
        return pos
    nxt = pos
    while nxt < limit and not text[nxt].isspace():
        nxt += 1
    while nxt < limit and text[nxt].isspace():
        nxt += 1
    return nxt if nxt < limit else pos


class FixedWindowChunkingStrategy(ChunkingStrategy):
    @property
    def name(self) -> str:
        return "fixed_window"

    def chunk(
        self,
        *,
        text: str,
        doc_id: str,
        source: str,
        chunk_size: int,
        chunk_overlap: int,
        page_number: int | None = None,
    ) -> list[RAGChunk]:
        normalized = text.strip()
        if not normalized:
            return []
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        chunks: list[RAGChunk] = []
        start = 0
        index = 0
        length = len(normalized)

        while start < length:
            while normalized[start].isspace():  # a window starts at a word (the text is stripped: one is ahead)
                start += 1
            end = start + chunk_size
            if 0 < length - end <= chunk_overlap:
                # Only a few characters are left after this window: the next window would be mostly the repeated
                # stretch, so they go into this one (at most chunk_size + chunk_overlap long) instead.
                end = length
            else:
                end = _cut(normalized, start, end)
            chunk_body = normalized[start:end].strip()
            if chunk_body:
                chunks.append(
                    RAGChunk(
                        doc_id=doc_id,
                        chunk_id=f"chunk-{index}",
                        source=source,
                        text=chunk_body,
                        page_number=page_number,
                        chunking_strategy=self.name,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                )
                index += 1
            if end >= length:
                # This window reached the end. Another one would hold only the end of this one: NCRC-UNICEF got
                # last passages "rnment." and "remain" this way (4 Nov), useless and close to short questions.
                break
            # The next window repeats the last chunk_overlap characters of this one, starting at a word.
            nxt = max(start + 1, end - chunk_overlap)
            start = _word_start(normalized, nxt, end) if chunk_overlap else nxt

        return chunks


def chunk_text(
    *,
    text: str,
    doc_id: str,
    source: str,
    chunk_size: int,
    chunk_overlap: int,
    page_number: int | None = None,
) -> list[RAGChunk]:
    return FixedWindowChunkingStrategy().chunk(
        text=text,
        doc_id=doc_id,
        source=source,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        page_number=page_number,
    )

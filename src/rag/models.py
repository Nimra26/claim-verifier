from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RAGChunk:
    doc_id: str
    chunk_id: str
    source: str
    text: str
    page_number: int | None = None
    chunking_strategy: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    embed_input: str | None = None  # how it was embedded: chunk_header.HEADER or BARE; None = BARE (before 3 Nov)


@dataclass(slots=True)
class RetrievedChunk:
    doc_id: str
    chunk_id: str
    source: str
    text: str
    score: float
    page_number: int | None = None
    rerank_score: float | None = None  # set by a reranker: 0..1 relevance to the question

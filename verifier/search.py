from __future__ import annotations

from src.rag.pipeline.services import RAGRetrievalService
from verifier.models import Passage


class EvidenceSearch:
    def __init__(self, retrieval: RAGRetrievalService, doc_ids: list[str], top_k: int) -> None:
        self._retrieval = retrieval
        self._doc_ids = doc_ids
        self._top_k = top_k

    async def search(self, query: str) -> list[Passage]:
        chunks = await self._retrieval.retrieve(query=query, top_k=self._top_k, doc_ids=self._doc_ids)
        return [
            Passage(
                id=f"{c.doc_id}:{c.chunk_id}",
                doc_id=c.doc_id,
                source=c.source,
                text=c.text,
                page=c.page_number,
                score=c.score,
            )
            for c in chunks
        ]

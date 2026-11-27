from __future__ import annotations

from abc import ABC, abstractmethod

from src.rag.models import RAGChunk, RetrievedChunk


class VectorStore(ABC):
    # Stores that can also search by keywords set this; retrieval then passes the query text along.
    supports_hybrid: bool = False

    @abstractmethod
    async def upsert_chunks(
        self,
        *,
        chunks: list[RAGChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Store chunks and their embeddings."""

    @abstractmethod
    async def similarity_search(
        self,
        *,
        query_embedding: list[float],
        top_k: int,
        doc_id: str | None = None,
        doc_ids: list[str] | None = None,
        query_text: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return top-k most similar chunks."""

    @abstractmethod
    async def list_chunks(self, *, doc_id: str) -> list[RAGChunk]:
        """Return all indexed chunks for a document."""

    @abstractmethod
    async def delete_by_doc_id(self, *, doc_id: str) -> None:
        """Delete all chunks for a given document id."""

    async def replace_chunks(
        self,
        *,
        doc_id: str,
        chunks: list[RAGChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Swap all of a document's chunks for a new set, so no chunk of the old version is left behind.

        This default is NOT atomic: between the delete and the upsert a search finds no chunk of the document,
        and if the upsert fails the document has no chunks until the next re-index. Stores that can do both in
        one transaction override it (PgVectorStore does).
        """
        check_replacement(doc_id=doc_id, chunks=chunks, embeddings=embeddings)
        await self.delete_by_doc_id(doc_id=doc_id)
        await self.upsert_chunks(chunks=chunks, embeddings=embeddings)


def check_replacement(*, doc_id: str, chunks: list[RAGChunk], embeddings: list[list[float]]) -> None:
    """Refuse a replacement that would delete the old chunks and then fail, or write chunks under another id."""
    if not doc_id:
        raise ValueError("replace_chunks needs a doc_id")
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")
    if any(chunk.doc_id != doc_id for chunk in chunks):
        raise ValueError("every chunk must belong to the document being replaced")

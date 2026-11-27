from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed_documents(self, texts: list[str], on_progress=None) -> list[list[float]]:
        """Embed a batch of documents/chunks. on_progress(done, total), when given, is told the parts (batches)
        done so far: (0, total) first, then after each part, so the Add documents page can show how far it is."""

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""

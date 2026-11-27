from __future__ import annotations

import asyncio
from typing import Any

from src.rag.embeddings.interface import EmbeddingProvider

try:
    from langchain_huggingface import HuggingFaceEmbeddings as _HuggingFaceEmbeddings
except ImportError:
    _HuggingFaceEmbeddings = None


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    def __init__(self, *, model_name: str) -> None:
        if _HuggingFaceEmbeddings is None:
            raise RuntimeError(
                "langchain-huggingface is not installed. Install it to use EMBEDDING_PROVIDER=huggingface."
            )
        self._embeddings = _HuggingFaceEmbeddings(model_name=model_name)

    async def embed_documents(self, texts: list[str], on_progress=None) -> list[list[float]]:
        if not texts:
            return []
        if on_progress is not None:
            on_progress(0, 1)  # one call for all of them, so the page can only say "started" and "done"
        vectors = await asyncio.to_thread(self._embeddings.embed_documents, texts)
        if on_progress is not None:
            on_progress(1, 1)
        return self._validate_vectors(vectors, context="documents")

    async def embed_query(self, text: str) -> list[float]:
        vector = await asyncio.to_thread(self._embeddings.embed_query, text)
        vectors = self._validate_vectors([vector], context="query")
        return vectors[0]

    @staticmethod
    def _validate_vectors(vectors: Any, *, context: str) -> list[list[float]]:
        if not isinstance(vectors, list):
            raise ValueError(f"HuggingFace embeddings returned invalid {context} vectors.")

        normalized: list[list[float]] = []
        for vec in vectors:
            if not isinstance(vec, list):
                raise ValueError(f"HuggingFace embeddings returned non-list vector for {context}.")
            normalized.append([float(value) for value in vec])
        return normalized

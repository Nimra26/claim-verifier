from __future__ import annotations

import os

from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from src.shared import resilience

from src.rag.embeddings.interface import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        organization: str | None = None,
    ) -> None:
        self._model = model
        # The library default was 2 retries and a ~10-minute timeout, so one stuck search could hold a question for
        # minutes (29 Oct review). Now 30 s (WORKSPACE_EMBED_TIMEOUT_S), and retries only for fast failures.
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            timeout=float(os.environ.get("WORKSPACE_EMBED_TIMEOUT_S", "30")),
            max_retries=0,
            # No proxy from the environment, even for localhost: it would carry our text off the machine (2 Nov).
            http_client=DefaultAsyncHttpxClient(trust_env=False),
        )
        # Adding a document sends every chunk to the model. In one call, a big PDF took bge-m3 on a laptop CPU past the
        # 30 s search timeout and the upload failed with HTTP 500 (eval night, 2 Nov: 12 of 13 files). So documents go
        # in batches, each with its own longer timeout; a question's single embedding keeps the 30 s one.
        self._batch = max(1, int(os.environ.get("WORKSPACE_EMBED_BATCH", "16")))
        self._doc_timeout = float(os.environ.get("WORKSPACE_EMBED_DOC_TIMEOUT_S", "120"))

    async def embed_documents(self, texts: list[str], on_progress=None) -> list[list[float]]:
        if not texts:
            return []
        client = self._client.with_options(timeout=self._doc_timeout)
        out: list[list[float]] = []
        parts = -(-len(texts) // self._batch)
        if on_progress is not None:
            on_progress(0, parts)
        for i in range(0, len(texts), self._batch):
            batch = texts[i : i + self._batch]
            response = await resilience.call(
                "model server", lambda b=batch: client.embeddings.create(model=self._model, input=b)
            )
            out.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
            if on_progress is not None:  # one call per batch: the page's "Indexing: 35%"
                on_progress(i // self._batch + 1, parts)
        return out

    async def embed_query(self, text: str) -> list[float]:
        response = await resilience.call("model server", lambda: self._client.embeddings.create(model=self._model, input=[text]))
        return response.data[0].embedding

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from uuid import uuid4

from src.rag.embeddings.interface import EmbeddingProvider
from src.rag.ingestion.chunker import ChunkingStrategyRegistry
from src.rag.ingestion.passage_cleanup import keep_tables_whole, tidy_passages, tidy_segments
from src.rag.ingestion.pdf_extractor import PDFExtractor
from src.rag.models import RAGChunk, RetrievedChunk
from src.rag.pipeline.chunk_header import BARE, HEADER, embed_input
from src.rag.reranker import Reranker
from src.rag.vectorstore.interface import VectorStore
from src.shared.progress import takes
from src.shared.tracing import TraceContext, chunk_metadata, trace_event


@dataclass(slots=True)
class IngestionResult:
    doc_id: str
    chunks_ingested: int
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int


@dataclass(slots=True)
class PDFIngestionResult:
    doc_id: str
    chunks_ingested: int
    pages_total: int
    pages_ingested: int
    skipped_pages: list[int]
    warnings: list[str]
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int


class RAGIngestionService:
    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        chunking_registry: ChunkingStrategyRegistry,
        default_chunking_strategy: str,
        chunk_size: int,
        chunk_overlap: int,
        pdf_extractor: PDFExtractor | None = None,
        pdf_max_pages: int = 300,
        chunk_headers: bool | Callable[[], bool] = False,
    ) -> None:
        """chunk_headers: embed each chunk with its document's name in front (chunk_header.py). A function is read
        at each embedding, so the app passes the Settings page's switch and a change applies to the next upload."""
        self._embedding_provider = embedding_provider
        self._chunk_headers = chunk_headers
        self._vector_store = vector_store
        self._chunking_registry = chunking_registry
        self._default_chunking_strategy = default_chunking_strategy
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._pdf_extractor = pdf_extractor
        self._pdf_max_pages = pdf_max_pages

    async def ingest_text(
        self,
        *,
        text: str,
        source: str | None = None,
        doc_id: str | None = None,
        chunking_strategy: str | None = None,
    ) -> IngestionResult:
        resolved_doc_id = doc_id or str(uuid4())
        chunks, embeddings, strategy_name = await self._chunk_and_embed(
            text=text,
            doc_id=resolved_doc_id,
            source=source or "inline-text",
            chunking_strategy=chunking_strategy,
        )
        await self._vector_store.upsert_chunks(chunks=chunks, embeddings=embeddings)

        return IngestionResult(
            doc_id=resolved_doc_id,
            chunks_ingested=len(chunks),
            chunking_strategy=strategy_name,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

    async def reindex_text(
        self,
        *,
        text: str,
        doc_id: str,
        source: str | None = None,
        chunking_strategy: str | None = None,
    ) -> IngestionResult:
        """Index a document's new text in place of its old chunks.

        Chunking and embedding run first; the old chunks are only replaced once the new ones are ready, so a
        failed embedding leaves the old version searchable.
        """
        chunks, embeddings, strategy_name = await self._chunk_and_embed(
            text=text,
            doc_id=doc_id,
            source=source or "inline-text",
            chunking_strategy=chunking_strategy,
        )
        if not chunks:
            raise ValueError("No chunks were generated from this text.")
        await self._vector_store.replace_chunks(doc_id=doc_id, chunks=chunks, embeddings=embeddings)

        return IngestionResult(
            doc_id=doc_id,
            chunks_ingested=len(chunks),
            chunking_strategy=strategy_name,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

    async def reindex_segments(
        self,
        *,
        segments: list[tuple[int | None, str]],
        doc_id: str,
        source: str,
        chunking_strategy: str | None = None,
        on_progress=None,
    ) -> IngestionResult:
        """Index a document given as (page number or None, text) pieces, in place of any old chunks.

        Used by the Add documents page: a PDF keeps its page numbers (the proof under an answer cites them), a
        slide deck its slide numbers, other files come as one piece. Like reindex_text, chunks are embedded before
        the old ones are swapped out, so a failure leaves the old version searchable. on_progress(done, total), when
        given, is passed to the embedding provider: the parts (batches) embedded so far, for the page's progress.
        """
        strategy = self._chunking_registry.resolve(chunking_strategy or self._default_chunking_strategy)
        chunks = self.passages(segments=segments, doc_id=doc_id, source=source, strategy=strategy)
        if not chunks:
            raise ValueError("No chunks were generated from this text.")
        embeddings = await self._embed_chunks(chunks, on_progress=on_progress)
        await self._vector_store.replace_chunks(doc_id=doc_id, chunks=chunks, embeddings=embeddings)
        return IngestionResult(
            doc_id=doc_id,
            chunks_ingested=len(chunks),
            chunking_strategy=strategy.name,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

    def strategy(self, name: str | None = None):
        """The chunking strategy by name; the default one for None or a name no longer registered."""
        try:
            return self._chunking_registry.resolve(name or self._default_chunking_strategy)
        except ValueError:
            return self._chunking_registry.resolve(self._default_chunking_strategy)

    def passages(self, *, segments: list[tuple[int | None, str]], doc_id: str, source: str,
                 strategy=None) -> list[RAGChunk]:
        """A document's passages, ready to embed: its (page or None, text) pieces tidied (front matter, running
        headers and footers: passage_cleanup.tidy_segments; a table kept with its title and header row in every part:
        keep_tables_whole), split, then tidied again (no thin or tiny passage, no
        repeat: tidy_passages), numbered chunk-0, chunk-1, ... Every way in (text, PDF, the Add documents page) and
        the re-index command (src/modules/workspace/reindex.py) make passages here, so they all clean the same way."""
        strategy = strategy or self.strategy()
        chunks: list[RAGChunk] = []
        for page_number, text in keep_tables_whole(tidy_segments(segments), chunk_size=self._chunk_size):
            chunks.extend(strategy.chunk(
                text=text,
                doc_id=doc_id,
                source=source,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                page_number=page_number,
            ))
        return tidy_passages(chunks, chunk_size=self._chunk_size)

    def _headers_on(self) -> bool:
        flag = self._chunk_headers
        return bool(flag() if callable(flag) else flag)

    async def _embed_chunks(self, chunks: list[RAGChunk], on_progress=None) -> list[list[float]]:
        """The ONE place document chunks are embedded: with the header when chunk_headers is on, else the bare text.
        Each chunk is stamped with how it was embedded (embed_input); its text is never changed."""
        headers = self._headers_on()
        for chunk in chunks:
            chunk.embed_input = HEADER if headers else BARE
        texts = [embed_input(c, headers=headers) for c in chunks]
        embed = self._embedding_provider.embed_documents
        if on_progress is None or not takes(embed, "on_progress"):  # only the Add documents page asks how far it is
            return await embed(texts)
        return await embed(texts, on_progress=on_progress)

    async def reembed_document(self, *, doc_id: str, dry_run: bool = False) -> int:
        """Re-embed a document's stored chunks that were embedded the other way (header vs bare) than the setting
        says now. Text, ids and pages stay as they are, so nothing is re-read or re-chunked. Returns how many vectors
        were written, or with dry_run how many would be (src/modules/workspace/reembed.py runs this for every
        document after the switch changes)."""
        update = getattr(self._vector_store, "update_embeddings", None)
        if update is None and not dry_run:
            raise RuntimeError("This vector store can't update embeddings in place; re-upload the documents instead.")
        mode = HEADER if self._headers_on() else BARE
        stale = [c for c in await self._vector_store.list_chunks(doc_id=doc_id) if (c.embed_input or BARE) != mode]
        if not stale or dry_run:
            return len(stale)
        embeddings = await self._embed_chunks(stale)
        return await update(chunks=stale, embeddings=embeddings)

    async def _chunk_and_embed(
        self,
        *,
        text: str,
        doc_id: str,
        source: str,
        chunking_strategy: str | None,
    ) -> tuple[list[RAGChunk], list[list[float]], str]:
        strategy = self._chunking_registry.resolve(chunking_strategy or self._default_chunking_strategy)
        chunks = self.passages(segments=[(None, text)], doc_id=doc_id, source=source, strategy=strategy)
        embeddings = await self._embed_chunks(chunks)
        return chunks, embeddings, strategy.name

    async def ingest_pdf(
        self,
        *,
        pdf_bytes: bytes,
        source: str | None = None,
        doc_id: str | None = None,
        chunking_strategy: str | None = None,
    ) -> PDFIngestionResult:
        if self._pdf_extractor is None:
            raise RuntimeError("PDF extractor is not configured.")

        resolved_doc_id = doc_id or str(uuid4())
        resolved_source = source or "uploaded-pdf"
        resolved_strategy_name = chunking_strategy or self._default_chunking_strategy
        strategy = self._chunking_registry.resolve(resolved_strategy_name)
        extraction = await self._pdf_extractor.extract(
            pdf_bytes=pdf_bytes,
            max_pages=self._pdf_max_pages,
        )

        if extraction.pages_ingested == 0:
            raise ValueError("No extractable pages were found in this PDF.")

        chunks = self.passages(segments=[(s.page_number, s.text) for s in extraction.segments],
                               doc_id=resolved_doc_id, source=resolved_source, strategy=strategy)

        if not chunks:
            raise ValueError("No chunks were generated from extracted PDF content.")

        embeddings = await self._embed_chunks(chunks)
        await self._vector_store.upsert_chunks(chunks=chunks, embeddings=embeddings)

        return PDFIngestionResult(
            doc_id=resolved_doc_id,
            chunks_ingested=len(chunks),
            pages_total=extraction.pages_total,
            pages_ingested=extraction.pages_ingested,
            skipped_pages=extraction.skipped_pages,
            warnings=extraction.warnings,
            chunking_strategy=strategy.name,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )


class RAGRetrievalService:
    RERANK_MAX_ATTEMPTS = 2
    RERANK_EXHAUSTED_ERROR = "Reranker failed after one retry."

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        default_top_k: int,
        prefetch_k: int,
        reranker: Reranker | None = None,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._default_top_k = default_top_k
        self._prefetch_k = prefetch_k
        self._reranker = reranker

    async def retrieve(
        self,
        *,
        query: str,
        top_k: int | None = None,
        doc_id: str | None = None,
        doc_ids: list[str] | None = None,
        trace_context: TraceContext | None = None,
    ) -> list[RetrievedChunk]:
        final_k = top_k or self._default_top_k
        if final_k < 1:
            raise ValueError("top_k must be >= 1")

        candidate_k = max(self._prefetch_k, final_k)
        trace_event(
            "retrieval.started",
            trace_context=trace_context,
            query=query,
            top_k=final_k,
            prefetch_k=candidate_k,
            reranker_enabled=self._reranker is not None,
            reranker_model=(
                getattr(self._reranker, "model_name", None)
                if self._reranker is not None
                else None
            ),
        )
        query_embedding = await self._embedding_provider.embed_query(query)
        search_kwargs: dict = {"query_embedding": query_embedding, "top_k": candidate_k, "doc_id": doc_id}
        if doc_ids is not None:
            # Only workspace search passes doc_ids, so other vector stores keep their old signature.
            search_kwargs["doc_ids"] = doc_ids
        if getattr(self._vector_store, "supports_hybrid", False) is True:
            search_kwargs["query_text"] = query
        candidates = await self._vector_store.similarity_search(**search_kwargs)
        trace_event(
            "retrieval.vector_search.completed",
            trace_context=trace_context,
            query=query,
            top_k=final_k,
            prefetch_k=candidate_k,
            candidate_count=len(candidates),
            candidates=chunk_metadata(candidates),
        )
        if not candidates:
            trace_event(
                "retrieval.completed",
                trace_context=trace_context,
                query=query,
                returned_count=0,
                reranker_status="disabled" if self._reranker is None else None,
                results=[],
            )
            return []
        if self._reranker is None:
            results = candidates[:final_k]
            trace_event(
                "retrieval.completed",
                trace_context=trace_context,
                query=query,
                returned_count=len(results),
                reranker_status="disabled",
                results=chunk_metadata(results),
            )
            return results

        for attempt in range(1, self.RERANK_MAX_ATTEMPTS + 1):
            try:
                reranked = await self._reranker.rerank(
                    query=query,
                    chunks=candidates,
                    top_n=final_k,
                )
                results = reranked[:final_k]
                reranker_status = (
                    "succeeded_first_try" if attempt == 1 else "succeeded_on_retry"
                )
                trace_event(
                    "retrieval.rerank.succeeded",
                    trace_context=trace_context,
                    query=query,
                    reranker_status=reranker_status,
                    reranker_attempt_count=attempt,
                    returned_count=len(results),
                    results=chunk_metadata(results),
                )
                trace_event(
                    "retrieval.completed",
                    trace_context=trace_context,
                    query=query,
                    returned_count=len(results),
                    reranker_status=reranker_status,
                    results=chunk_metadata(results),
                )
                return results
            except Exception as exc:
                if attempt >= self.RERANK_MAX_ATTEMPTS:
                    trace_event(
                        "retrieval.rerank.failed",
                        trace_context=trace_context,
                        query=query,
                        reranker_status="failed",
                        reranker_attempt_count=attempt,
                        error=str(exc),
                    )
                    raise RuntimeError(self.RERANK_EXHAUSTED_ERROR) from exc
                trace_event(
                    "retrieval.rerank.retry",
                    trace_context=trace_context,
                    query=query,
                    reranker_attempt_count=attempt,
                    error=str(exc),
                )

        raise RuntimeError(self.RERANK_EXHAUSTED_ERROR)

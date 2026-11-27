from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from src.rag.embeddings.interface import EmbeddingProvider
from src.rag.ingestion.chunking import (
    ChunkingStrategyRegistry,
    FixedWindowChunkingStrategy,
    RecursiveSemanticChunkingStrategy,
)
from src.rag.pipeline.services import RAGIngestionService, RAGRetrievalService
from src.rag.vectorstore.interface import VectorStore
from verifier.config import VerifierSettings

# DDL copied from alembic revision b7c1d2e3f4a5 (add rag_chunks pgvector), made idempotent with IF NOT EXISTS.
_SCHEMA_STATEMENTS = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    """
    CREATE TABLE IF NOT EXISTS rag_chunks (
        id text PRIMARY KEY,
        doc_id text NOT NULL,
        chunk_id text NOT NULL,
        source text NOT NULL,
        page_number integer,
        text text NOT NULL,
        embedding vector(1024) NOT NULL,
        meta jsonb NOT NULL DEFAULT '{}'::jsonb,
        tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', coalesce(text, ''))) STORED,
        created_at timestamptz NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_rag_chunks_doc_id ON rag_chunks (doc_id)",
    "CREATE INDEX IF NOT EXISTS ix_rag_chunks_embedding ON rag_chunks USING hnsw (embedding vector_cosine_ops)",
    "CREATE INDEX IF NOT EXISTS ix_rag_chunks_tsv ON rag_chunks USING gin (tsv)",
]


async def ensure_schema(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        for statement in _SCHEMA_STATEMENTS:
            await conn.execute(text(statement))


@dataclass
class Store:
    ingestion: RAGIngestionService
    retrieval: RAGRetrievalService
    vector_store: VectorStore
    embedding_provider: EmbeddingProvider | None = None


def build_store(
    settings: VerifierSettings,
    *,
    embedding_provider: EmbeddingProvider | None = None,
    vector_store: VectorStore | None = None,
) -> Store:
    if embedding_provider is None:
        from src.infrastructure.llm.openai_embeddings import OpenAIEmbeddingProvider

        embedding_provider = OpenAIEmbeddingProvider(
            api_key="ollama",
            model=settings.embed_model,
            base_url=f"{settings.ollama_url}/v1",
        )
    if vector_store is None:
        from src.infrastructure.vector_db.pgvector_store import PgVectorStore

        vector_store = PgVectorStore(database_url=settings.database_url)
    registry = ChunkingStrategyRegistry([FixedWindowChunkingStrategy(), RecursiveSemanticChunkingStrategy()])
    ingestion = RAGIngestionService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        chunking_registry=registry,
        default_chunking_strategy="recursive_semantic",
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    retrieval = RAGRetrievalService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        default_top_k=settings.top_k,
        prefetch_k=2 * settings.top_k,
        reranker=None,
    )
    return Store(ingestion=ingestion, retrieval=retrieval, vector_store=vector_store,
                 embedding_provider=embedding_provider)


async def index_documents(store: Store, texts: dict[str, str], titles: dict[str, str]) -> None:
    for doc_id, body in texts.items():
        await store.vector_store.delete_by_doc_id(doc_id=doc_id)
        await store.ingestion.ingest_text(text=body, source=titles[doc_id], doc_id=doc_id)

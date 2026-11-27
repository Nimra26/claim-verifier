"""Postgres + pgvector vector store with hybrid (meaning + keyword) search.

One database holds users, documents and chunk embeddings, so backups, exports and access rules live in one place.
Search runs two queries inside the allowed documents and merges them with reciprocal-rank fusion (RRF):
  - meaning: cosine distance on the embedding (HNSW index, iterative scan so filtered searches still return k rows)
  - keyword: Postgres full-text on the chunk text ('simple' config, so Urdu and English tokens both work)
Returned scores are always the cosine similarity (0..1, higher is closer), whichever list found the chunk.

Prefer a document named in the question (setting title_boost, OFF by default, 3 Nov): when the question shares rare
words with a document's title (its file name), that document's chunks get a bounded bonus in the fused ranking. Only
the order changes, never the scores, and only titles of the allowed documents are read (see named_documents).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Callable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from src.rag.models import RAGChunk, RetrievedChunk
from src.rag.passage_filter import NOT_THIN_SQL_BOUND
from src.rag.pipeline.chunk_header import document_title
from src.rag.vectorstore import VectorStore
from src.rag.vectorstore.interface import check_replacement

RRF_K = 60  # standard RRF constant (Cormack et al., 2009)
DIMENSIONS = 1024  # rag_chunks.embedding is vector(1024): bge-m3
TOKEN = re.compile(r"[^\W_]{2,}", re.UNICODE)


# Question words that say nothing about the topic. The index uses the 'simple' config (no stop-word list, so
# Urdu works), so they are dropped here: with them, "What GPUs should we ask sky47 for?" ranked grade-6 science
# exercises first on "what/should/for" and pushed the right sky47 passage out of the top 5 (28 Oct).
STOPWORDS = frozenset("""
a an the and or but if of to in on at by for from with about into over under than then so as is are was were be
been being am do does did doing have has had having i me my we our us you your he she it its they them their this
that these those there here what which who whom whose when where why how can could should would will shall may
might must not no yes any all some each every more most other such only own same too very just also please tell
give show explain list mean meaning
kya kia hai hain hy ha tha thi the ho hota hoti hote ka ki ke ko se mein main me par pe aur ya bhi to toh nahi
nahin kaise kese kaisay kyun kyon kahan kab kaun konsa kis kitne kitna yeh ye woh wo is us un in hum ap aap
batao bataen bataein matlab liye chahiye karna karte kar wala wali
کیا ہے ہیں تھا تھی تھے ہو ہوتا ہوتی ہوتے کا کی کے کو سے میں پر اور یا بھی تو نہیں کیسے کیوں کہاں کب کون کونسا
کس کتنے کتنا یہ وہ اس ان ہم آپ بتائیں مطلب لیے چاہیے کرنا کرتے کر والا والی
""".split())


def topic_words(text_: str | None) -> list[str]:
    """Topic words, lowercased, in order, once each: what both the keyword search and the title match use."""
    words = [w.lower() for w in TOKEN.findall(text_ or "")]
    return list(dict.fromkeys(w for w in words if w not in STOPWORDS))


def keyword_query(text_: str) -> str | None:
    """OR of the question's topic words for to_tsquery. Only letters/digits survive, so the string is safe to pass.

    websearch_to_tsquery ANDs every word, so a whole question rarely matched any chunk (28 Oct review: 7 of 8
    test questions returned nothing); OR plus ts_rank ordering rewards chunks that share the most topic words.
    Returns None when the question has no topic words; the search is then meaning-only.
    """
    words = topic_words(text_)[:24]
    return " | ".join(words) if words else None


# --- prefer a document named in the question (title_boost) ----------------------------------------------------------
# Public eval Q5 (3 Nov): "... out of school according to the 2023 Census ..." The answer is a row of the file
# PBS_Census-2023_Table12-..., but "Census 2023" is only in that file's name and first chunk, so the ITA ASER report,
# which shares more of the question's words, ranked first every time. Putting the name into every chunk's embedding
# (chunk_headers) gave no gain. This is the established fix of fielded search (BM25F; Elasticsearch / OpenSearch
# multi_match with a title field): a match in the title counts on its own, weighted by how rare the word is in titles.
#
# Bounded by design: the bonus is a third rank list over the NAMED documents, TITLE_BOOST_WEIGHT / (RRF_K + rank),
# added to each of that document's chunks the two searches already found. At most half of one list's first place
# (0.5 / 61), so it settles near ties, but a passage found first by both searches (2 / 61) always beats a named
# document's passage that only one search found (at most 1 / 61 + 0.5 / 61). It never adds chunks and never changes
# the question's embedding or a chunk's score, so the refusal floor reads the same numbers.
TITLE_BOOST_WEIGHT = 0.5
# A title is "named" when its shared words are worth at least one word that is in no other title. Each word counts
# ln(N / df) / ln(N) over the N titles the person may search: 1 for a word in one title, 0 for a word in all of them.
# Common words alone stay below it: in the 13-file public demo set "pakistan" is in 6 titles (0.30), "2023" and
# "education" in 3 (0.57), "national" in 2 (0.73), while "census" is in one (1.0). Decided by the titles, no word list.
TITLE_MIN_MATCH = 1.0


def title_words(source: str | None) -> set[str]:
    """A document's title words: its file name without folder or extension (chunk_header.document_title), split into
    words, stop words dropped, and numbers shorter than 4 digits dropped (table, part and chapter numbers like "12", or
    the "26" of "2025-26", would match any quantity in a question; years stay)."""
    title = document_title(source)
    if not title:
        return set()
    return {w for w in topic_words(title) if not (w.isdigit() and len(w) < 4)}


def named_documents(question: str | None, titles: dict[str, str]) -> dict[str, float]:
    """{doc_id: match} for the documents whose title the question names (match >= TITLE_MIN_MATCH, see above).

    titles maps doc_id -> source for the documents this person may search, and ONLY those: how rare a word is is
    counted among them, so a title they can't see changes nothing. With fewer than two titles there is nothing to
    prefer, so nothing is named.
    """
    words = set(topic_words(question))
    if not words or len(titles) < 2:
        return {}
    per_doc = {doc: title_words(source) for doc, source in titles.items()}
    df = Counter(w for ws in per_doc.values() for w in ws)
    unit = math.log(len(per_doc))
    named = {}
    for doc, ws in per_doc.items():
        match = sum(math.log(len(per_doc) / df[w]) for w in ws & words) / unit
        if match >= TITLE_MIN_MATCH - 1e-9:
            named[doc] = round(match, 6)
    return named


def _vec(values: list[float]) -> str:
    return "[" + ",".join(f"{float(v):.7f}" for v in values) + "]"


def rrf_fuse(ranked_lists: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal-rank fusion: each list adds 1/(k + rank) for every id it contains."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, key in enumerate(ranked, 1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return scores


class PgVectorStore(VectorStore):
    supports_hybrid = True

    def __init__(self, *, database_url: str, hybrid: bool = True, engine: AsyncEngine | None = None,
                 dimensions: int = DIMENSIONS, title_boost: bool | Callable[[], bool] = False) -> None:
        """title_boost: prefer a document named in the question (above). A function is read at each search, so the
        app passes the Settings page's switch and a change applies to the next question."""
        self._engine = engine or create_async_engine(database_url, echo=False, pool_pre_ping=True)
        self._hybrid = hybrid
        self._dimensions = dimensions
        self._title_boost = title_boost

    def _title_boost_on(self) -> bool:
        flag = self._title_boost
        try:
            return (flag() if callable(flag) else flag) is True
        except Exception:  # a switch that can't be read means off: the search as before
            return False

    async def upsert_chunks(self, *, chunks: list[RAGChunk], embeddings: list[list[float]]) -> None:
        rows = self._rows(chunks, embeddings)
        if not rows:
            return
        async with self._engine.begin() as conn:
            await conn.execute(text(self._UPSERT), rows)

    async def replace_chunks(self, *, doc_id: str, chunks: list[RAGChunk], embeddings: list[list[float]]) -> None:
        # Delete and insert in ONE transaction: a search sees the old version or the new one, never a mix, and no
        # chunk id of a longer old version survives. If the insert fails, the old chunks stay.
        check_replacement(doc_id=doc_id, chunks=chunks, embeddings=embeddings)
        rows = self._rows(chunks, embeddings)
        async with self._engine.begin() as conn:
            await conn.execute(text("DELETE FROM rag_chunks WHERE doc_id = :d"), {"d": doc_id})
            if rows:
                await conn.execute(text(self._UPSERT), rows)

    _UPSERT = (
        "INSERT INTO rag_chunks (id, doc_id, chunk_id, source, page_number, text, embedding, meta) "
        "VALUES (:id, :doc_id, :chunk_id, :source, :page_number, :text, CAST(:embedding AS vector), "
        "CAST(:meta AS jsonb)) "
        "ON CONFLICT (id) DO UPDATE SET source = EXCLUDED.source, page_number = EXCLUDED.page_number, "
        "text = EXCLUDED.text, embedding = EXCLUDED.embedding, meta = EXCLUDED.meta"
    )

    def _rows(self, chunks: list[RAGChunk], embeddings: list[list[float]]) -> list[dict]:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if embeddings and len(embeddings[0]) != self._dimensions:
            raise ValueError(
                f"Embedding has {len(embeddings[0])} dimensions but rag_chunks stores {self._dimensions} (bge-m3). "
                "Use EMBEDDING_MODEL=bge-m3 or migrate the table for the new model."
            )
        return [
            {
                "id": f"{c.doc_id}:{c.chunk_id}", "doc_id": c.doc_id, "chunk_id": c.chunk_id, "source": c.source,
                "page_number": c.page_number, "text": c.text, "embedding": _vec(e),
                "meta": json.dumps({"chunking_strategy": c.chunking_strategy, "chunk_size": c.chunk_size,
                                    "chunk_overlap": c.chunk_overlap, "embed_input": c.embed_input}),
            }
            for c, e in zip(chunks, embeddings)
        ]

    async def similarity_search(
        self,
        *,
        query_embedding: list[float],
        top_k: int,
        doc_id: str | None = None,
        doc_ids: list[str] | None = None,
        query_text: str | None = None,
    ) -> list[RetrievedChunk]:
        if doc_ids is not None and not doc_ids:
            return []  # workspace search with nothing visible
        if doc_ids is None and not doc_id:
            raise ValueError("pgvector search needs doc_id or doc_ids; an unfiltered search would cross users")
        allowed = doc_ids if doc_ids is not None else ([doc_id] if doc_id else None)
        # Thin passages ("o", "not.", a page number) are never returned (passage_filter.py, 4 Nov): stored ones from
        # before that change stay out of answers without a re-index. Inside the SQL, so the k rows are all real ones.
        where = ("WHERE doc_id = ANY(:allowed) AND " if allowed is not None else "WHERE ") + NOT_THIN_SQL_BOUND
        params = {"q": _vec(query_embedding), "k": top_k, "allowed": allowed}
        cols = "id, doc_id, chunk_id, source, page_number, text, 1 - (embedding <=> CAST(:q AS vector)) AS sim"
        async with self._engine.connect() as conn:
            # Filtered HNSW searches would otherwise return fewer than k rows for small groups (pgvector >= 0.8).
            await conn.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
            vec_rows = (await conn.execute(
                text(f"SELECT {cols} FROM rag_chunks {where} ORDER BY embedding <=> CAST(:q AS vector) LIMIT :k"),
                params,
            )).mappings().all()
            # relaxed_order returns rows roughly by distance; put them in exact order before ranking.
            vec_rows = sorted(vec_rows, key=lambda r: -float(r["sim"]))
            kw_rows = []
            tsq = keyword_query(query_text) if self._hybrid else None
            if tsq:
                kw_rows = (await conn.execute(
                    text(
                        f"SELECT {cols}, ts_rank_cd(tsv, to_tsquery('simple', :t)) AS kw FROM rag_chunks "
                        f"{where} AND tsv @@ to_tsquery('simple', :t) "
                        "ORDER BY kw DESC LIMIT :k"
                    ),
                    {**params, "t": tsq},
                )).mappings().all()
            titles: dict[str, str] = {}
            if self._title_boost_on() and len(allowed) > 1 and topic_words(query_text):
                # The same allowed filter as both searches: titles of documents this person can't search are never read.
                titles = dict((await conn.execute(
                    text("SELECT DISTINCT ON (doc_id) doc_id, source FROM rag_chunks WHERE doc_id = ANY(:allowed) "
                         "ORDER BY doc_id"),
                    {"allowed": allowed},
                )).all())
            await conn.rollback()

        by_id = {r["id"]: r for r in [*vec_rows, *kw_rows]}
        fused = rrf_fuse([[r["id"] for r in vec_rows], [r["id"] for r in kw_rows]])
        named = named_documents(query_text, titles) if titles else {}
        for rank, doc in enumerate(sorted(named, key=lambda d: (-named[d], d)), 1):
            bonus = TITLE_BOOST_WEIGHT / (RRF_K + rank)
            for key in fused:
                if by_id[key]["doc_id"] == doc:
                    fused[key] += bonus
        order = sorted(fused, key=lambda key: (-fused[key], -float(by_id[key]["sim"])))[:top_k]
        return [
            RetrievedChunk(
                doc_id=by_id[key]["doc_id"], chunk_id=by_id[key]["chunk_id"], source=by_id[key]["source"],
                text=by_id[key]["text"], score=round(float(by_id[key]["sim"]), 6), page_number=by_id[key]["page_number"],
            )
            for key in order
        ]

    async def list_chunks(self, *, doc_id: str) -> list[RAGChunk]:
        if not doc_id:
            return []
        async with self._engine.connect() as conn:
            rows = (await conn.execute(
                text("SELECT doc_id, chunk_id, source, page_number, text, meta FROM rag_chunks WHERE doc_id = :d"),
                {"d": doc_id},
            )).mappings().all()
        out = []
        for r in rows:
            meta = r["meta"] or {}
            out.append(RAGChunk(doc_id=r["doc_id"], chunk_id=r["chunk_id"], source=r["source"], text=r["text"],
                                page_number=r["page_number"], chunking_strategy=meta.get("chunking_strategy"),
                                chunk_size=meta.get("chunk_size"), chunk_overlap=meta.get("chunk_overlap"),
                                embed_input=meta.get("embed_input")))
        return sorted(out, key=lambda c: (c.page_number or 0, c.chunk_id))

    async def update_embeddings(self, *, chunks: list[RAGChunk], embeddings: list[list[float]]) -> int:
        """Write new vectors for chunks already stored (RAGIngestionService.reembed_document). A row is updated only
        while its text, source and page are still the ones that were embedded: if the document was re-uploaded in the
        meantime (that lock is in the app, not here), its new chunks are left alone. Returns the rows updated."""
        rows = self._rows(chunks, embeddings)
        if not rows:
            return 0
        updated = 0
        async with self._engine.begin() as conn:
            for row in rows:
                result = await conn.execute(text(
                    "UPDATE rag_chunks SET embedding = CAST(:embedding AS vector), "
                    "meta = meta || jsonb_build_object('embed_input', CAST(:stamp AS text)) "
                    "WHERE id = :id AND text = :text AND source = :source "
                    "AND page_number IS NOT DISTINCT FROM CAST(:page_number AS int)"
                ), {"embedding": row["embedding"], "stamp": json.loads(row["meta"])["embed_input"], "id": row["id"],
                    "text": row["text"], "source": row["source"], "page_number": row["page_number"]})
                updated += result.rowcount or 0
        return updated

    async def delete_by_doc_id(self, *, doc_id: str) -> None:
        if not doc_id:
            return
        async with self._engine.begin() as conn:
            await conn.execute(text("DELETE FROM rag_chunks WHERE doc_id = :d"), {"d": doc_id})

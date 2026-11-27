from __future__ import annotations

import hashlib
import math
import re

from src.rag.embeddings.interface import EmbeddingProvider
from src.rag.models import RAGChunk, RetrievedChunk
from src.rag.vectorstore.interface import VectorStore

DIM = 1024


def _embed(text: str) -> list[float]:
    vec = [0.0] * DIM
    for word in re.findall(r"\w+", text.lower()):
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vec[h % DIM] += 1.0
    return vec


class FakeEmbeddings(EmbeddingProvider):
    async def embed_documents(self, texts, on_progress=None):
        if on_progress:
            on_progress(0, 1)
        out = [_embed(t) for t in texts]
        if on_progress:
            on_progress(1, 1)
        return out

    async def embed_query(self, text):
        return _embed(text)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class MemoryVectorStore(VectorStore):
    supports_hybrid = False

    def __init__(self) -> None:
        self.rows: list[tuple[RAGChunk, list[float]]] = []

    async def upsert_chunks(self, *, chunks, embeddings):
        for chunk, emb in zip(chunks, embeddings):
            self.rows = [r for r in self.rows if (r[0].doc_id, r[0].chunk_id) != (chunk.doc_id, chunk.chunk_id)]
            self.rows.append((chunk, emb))

    async def delete_by_doc_id(self, *, doc_id):
        self.rows = [r for r in self.rows if r[0].doc_id != doc_id]

    async def list_chunks(self, *, doc_id):
        return [c for c, _ in self.rows if c.doc_id == doc_id]

    async def similarity_search(self, *, query_embedding, top_k, doc_id=None, doc_ids=None, query_text=None):
        scored = []
        for chunk, emb in self.rows:
            if doc_id is not None and chunk.doc_id != doc_id:
                continue
            if doc_ids is not None and chunk.doc_id not in doc_ids:
                continue
            scored.append(
                RetrievedChunk(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    text=chunk.text,
                    score=_cosine(query_embedding, emb),
                    page_number=chunk.page_number,
                )
            )
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# FakeReasoner
# ---------------------------------------------------------------------------
from verifier.models import Claim as _Claim, Judgment as _Judgment, Stance as _Stance  # noqa: E402
from verifier.reasoner import DocFacts as _DocFacts, ExtractedClaims as _ExtractedClaims  # noqa: E402


class FakeReasoner:
    """Scripted Reasoner for tests. Every call is recorded in `.calls` as (method_name, args_tuple).

    Each constructor argument is optional and may be a dict or a callable:
      doc_facts      dict keyed by title, or callable(title, text, other_titles) -> DocFacts
                     (default: DocFacts(date=None, replaces=[]))
      extract_claims dict keyed by doc.id, or callable(question, doc, text) -> ExtractedClaims
                     (default: no claims, dropped_non_factual=0)
      group_claims   dict {claim_id: subject} (used as-is), or callable(question, claims) -> dict
                     (default: {})
      judge          dict keyed by claim.id, or callable(claim, passages) -> Judgment
                     (default: every passage "unrelated", states_claim=True, confidence "high")
      reword         dict keyed by claim.id, or callable(claim, tried) -> str
                     (default: f"{claim.text} (reworded {len(tried)})")
      write_answer   str, or callable(question, claims, verdicts, caveat) -> str
                     (default: "Answer based on N claim(s).")
    A dict miss falls back to the default. Values for judge may also be a list, consumed one per call
    (the last item repeats), to script re-check sequences.
    """

    def __init__(self, doc_facts=None, extract_claims=None, group_claims=None, judge=None, reword=None, write_answer=None):
        self._s = {
            "doc_facts": doc_facts,
            "extract_claims": extract_claims,
            "group_claims": group_claims,
            "judge": judge,
            "reword": reword,
            "write_answer": write_answer,
        }
        self._judge_n: dict[str, int] = {}
        self.calls: list[tuple[str, tuple]] = []

    def _lookup(self, name, key, args):
        s = self._s[name]
        if callable(s):
            return True, s(*args)
        if isinstance(s, dict):
            if key in s:
                return True, s[key]
            return False, None
        if s is not None:
            return True, s
        return False, None

    async def doc_facts(self, title, text, other_titles):
        self.calls.append(("doc_facts", (title, text, other_titles)))
        ok, v = self._lookup("doc_facts", title, (title, text, other_titles))
        return v if ok else _DocFacts()

    async def extract_claims(self, question, doc, text):
        self.calls.append(("extract_claims", (question, doc, text)))
        ok, v = self._lookup("extract_claims", doc.id, (question, doc, text))
        return v if ok else _ExtractedClaims(claims=[], dropped_non_factual=0)

    async def group_claims(self, question, claims):
        self.calls.append(("group_claims", (question, claims)))
        s = self._s["group_claims"]
        if isinstance(s, dict):
            return dict(s)
        if callable(s):
            return s(question, claims)
        return {}

    async def judge(self, claim, passages):
        self.calls.append(("judge", (claim, passages)))
        s = self._s["judge"]
        if isinstance(s, dict) and claim.id in s:
            s = s[claim.id]
        elif isinstance(s, dict):
            s = None
        if isinstance(s, list):
            n = self._judge_n.get(claim.id, 0)
            self._judge_n[claim.id] = n + 1
            s = s[min(n, len(s) - 1)]
        if callable(s):
            return s(claim, passages)
        if s is not None:
            return s
        return _Judgment(
            stances=[_Stance(passage_id=p.id, stance="unrelated") for p in passages],
            states_claim=True,
            confidence="high",
            reason="default fake judgment",
        )

    async def reword(self, claim, tried):
        self.calls.append(("reword", (claim, tried)))
        ok, v = self._lookup("reword", claim.id, (claim, tried))
        return v if ok else f"{claim.text} (reworded {len(tried)})"

    async def write_answer(self, question, claims, verdicts, caveat):
        self.calls.append(("write_answer", (question, claims, verdicts, caveat)))
        ok, v = self._lookup("write_answer", None, (question, claims, verdicts, caveat))
        return v if ok else f"Answer based on {len(claims)} claim(s)."

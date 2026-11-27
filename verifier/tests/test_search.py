import asyncio

from verifier.config import VerifierSettings
from verifier.search import EvidenceSearch
from verifier.store import build_store, index_documents
from verifier.tests.fakes import FakeEmbeddings, MemoryVectorStore

TEXTS = {
    "a": "The quarterly budget was approved by the board in March.",
    "b": "Zeppelin deliveries to the northern depot begin in June.",
}
TITLES = {"a": "Budget Memo", "b": "Logistics Plan"}


def _store():
    return build_store(VerifierSettings(), embedding_provider=FakeEmbeddings(), vector_store=MemoryVectorStore())


def test_search_maps_chunks_to_passages():
    async def go():
        store = _store()
        await index_documents(store, TEXTS, TITLES)
        return await EvidenceSearch(store.retrieval, ["a", "b"], 8).search("zeppelin")

    passages = asyncio.run(go())
    assert passages[0].doc_id == "b"
    assert passages[0].id.startswith("b:")
    assert passages[0].source == "Logistics Plan"


def test_search_limited_to_doc_ids():
    async def go():
        store = _store()
        await index_documents(store, TEXTS, TITLES)
        return await EvidenceSearch(store.retrieval, ["a"], 8).search("zeppelin")

    passages = asyncio.run(go())
    assert passages and {p.doc_id for p in passages} == {"a"}


def test_index_replaces_previous_chunks():
    async def go():
        store = _store()
        await index_documents(store, TEXTS, TITLES)
        first = len(await store.vector_store.list_chunks(doc_id="a"))
        await index_documents(store, TEXTS, TITLES)
        return first, len(await store.vector_store.list_chunks(doc_id="a"))

    first, second = asyncio.run(go())
    assert first >= 1 and first == second

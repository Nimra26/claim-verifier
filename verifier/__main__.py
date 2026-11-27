from __future__ import annotations

import verifier.appenv  # noqa: F401  - must run before any src import (see verifier/appenv.py)

import argparse
import asyncio
import socket
import sys
import uuid
from pathlib import Path

import yaml
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from verifier.config import VerifierSettings
from verifier.graph import build_graph
from verifier.manifest import load_manifest
from verifier.report import write_report
from verifier.search import EvidenceSearch
from verifier.store import Store, build_store, ensure_schema, index_documents

MAX_TRIES = 3
DB_DOWN = "Could not reach the document database. Start it first (see README.md)."
OLLAMA_DOWN = "Could not reach the embedding model (Ollama). Start Ollama and run: ollama pull bge-m3"


class _Exit(Exception):
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message


def _chain(exc: BaseException):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


_NETWORK_ERROR_NAMES = {
    # openai / anthropic / httpx
    "APIConnectionError", "APITimeoutError", "ConnectError", "ConnectTimeout",
    # asyncpg
    "CannotConnectNowError", "ConnectionDoesNotExistError", "ConnectionFailureError", "PostgresConnectionError",
}


def _is_connection_error(exc: BaseException) -> bool:
    """Network errors only (M1): a PermissionError or other OSError is not "service unreachable"."""
    return any(isinstance(e, (ConnectionError, socket.gaierror, TimeoutError))
               or type(e).__name__ in _NETWORK_ERROR_NAMES for e in _chain(exc))


def _is_ollama_error(exc: BaseException) -> bool:
    return any(type(e).__name__ in {"APIConnectionError", "ConnectError", "ConnectTimeout", "APITimeoutError"}
               or type(e).__module__.split(".")[0] in {"openai", "httpx"} for e in _chain(exc))


def _load_answers(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        raise _Exit(2, f"Answers file not found: {path}")
    except yaml.YAMLError:
        raise _Exit(2, f"Answers file is not valid YAML: {path}")
    if not isinstance(raw, dict):
        raise _Exit(2, f"Answers file must map question id to answer: {path}")
    return {str(k): str(v) for k, v in raw.items()}


def _pick(reply: str, choices: list[str]) -> str | None:
    reply = reply.strip()
    if reply.isdigit() and 1 <= int(reply) <= len(choices):
        return choices[int(reply) - 1]
    return reply if reply in choices else None


def _answer(question: dict, answers: dict[str, str], input_fn, unattended: bool = False) -> str:
    choices = list(question["choices"])
    given = answers.get(question["id"])
    if given is not None:
        if given in choices:
            return given
        print(f"The answer in the answers file for {question['id']} is not one of the choices; "
              + ("using the last choice instead." if unattended else "asking instead."))
    if unattended:
        # M5: the last choice is the non-committal one ("neither", "can't tell"; "no" for a guessed date).
        print(f"Unattended: {question['id']} ({question['text']}) answered with: {choices[-1]}")
        return choices[-1]
    print()
    print(question["text"])
    for i, choice in enumerate(choices, 1):
        print(f"{i}. {choice}")
    for _ in range(MAX_TRIES):
        picked = _pick(input_fn("Your answer (number or text): "), choices)
        if picked is not None:
            return picked
        print("Please type one of the numbers or the exact choice text.")
    print(f"No valid answer after {MAX_TRIES} tries; using the last choice: {choices[-1]}")
    return choices[-1]


async def _run(args, reasoner, store, input_fn, *, ask=None, say=print) -> int:
    """`ask(question) -> choice` replaces the terminal prompt (used by the web page); `say` receives progress lines."""
    folder = Path(args.folder)
    if not folder.is_dir():
        raise _Exit(2, f"Folder not found: {folder}")
    if not (folder / "manifest.yaml").is_file():
        raise _Exit(2, f"No manifest.yaml in {folder}")
    try:
        manifest, texts = load_manifest(folder)
    except (FileNotFoundError, ValueError, KeyError) as e:
        raise _Exit(2, f"Could not read the manifest: {e}")
    answers = _load_answers(args.answers)

    settings = VerifierSettings()
    own_reasoner = reasoner is None
    if reasoner is None:
        from verifier.reasoner import make_reasoner

        try:
            reasoner = make_reasoner(settings.backend, settings.model)
        except RuntimeError as e:
            raise _Exit(2, str(e))

    engine = None
    try:
        if store is None:
            from sqlalchemy.ext.asyncio import create_async_engine
            from src.infrastructure.vector_db.pgvector_store import PgVectorStore

            engine = create_async_engine(settings.database_url, pool_pre_ping=True)
            store = build_store(settings, vector_store=PgVectorStore(database_url=settings.database_url, engine=engine))
            try:
                await ensure_schema(engine)
            except Exception as e:
                if _is_connection_error(e):
                    raise _Exit(3, DB_DOWN)
                raise
        titles = {d.id: d.title for d in manifest.documents}
        say(f"Indexing {len(texts)} documents...")
        try:
            await index_documents(store, texts, titles)
        except Exception as e:
            if _is_ollama_error(e):
                raise _Exit(3, OLLAMA_DOWN)
            if _is_connection_error(e):
                raise _Exit(3, DB_DOWN)
            raise
        search = EvidenceSearch(store.retrieval, list(texts), settings.top_k)
        graph = build_graph(reasoner, search, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        payload = {"question": manifest.question,
                   "documents": {d.id: d for d in manifest.documents}, "texts": texts}
        say("Checking claims...")
        result = await graph.ainvoke(payload, config)
        while result.get("__interrupt__"):
            question = result["__interrupt__"][0].value
            result = await graph.ainvoke(
                Command(resume=ask(question) if ask else _answer(question, answers, input_fn, args.unattended)), config)
        for line in result["report"].failures:
            say(f"Note: {line}")
        report = result["report"]
        try:
            md_path, json_path = write_report(report, folder)
        except OSError:
            raise _Exit(2, f"Could not write the report to {folder}")
        say(f"Report written: {md_path}")
        say(f"Report written: {json_path}")
        if not report.completed:
            say(report.answer.text)
            return 4
        return 0
    finally:
        if store is not None:
            await _close_embeddings(store)
        if own_reasoner and hasattr(reasoner, "aclose"):
            await reasoner.aclose()
        if engine is not None:
            await engine.dispose()


async def _close_embeddings(store: Store) -> None:
    """M6: close the embedding provider's HTTP client if it has one (src/ is not changed for this)."""
    provider = getattr(store, "embedding_provider", None)
    close = getattr(provider, "aclose", None) or getattr(provider, "close", None)
    if close is None:
        close = getattr(getattr(provider, "_client", None), "close", None)
    if close is None:
        return
    try:
        result = close()
        if asyncio.iscoroutine(result):
            await result
    except Exception:  # noqa: BLE001 - closing is best effort
        pass


def main(argv: list[str], *, reasoner=None, store: Store | None = None, input_fn=input) -> int:
    parser = argparse.ArgumentParser(prog="verifier", description="Check claims in a folder of documents.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="verify the documents in a folder")
    run.add_argument("folder")
    run.add_argument("--answers", help="YAML file mapping question id to answer")
    run.add_argument("--unattended", action="store_true",
                     help="never prompt: a question not in --answers gets its last (non-committal) choice")
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_run(args, reasoner, store, input_fn))
    except _Exit as e:
        print(e.message)
        return e.code
    except Exception as e:
        if _is_connection_error(e):
            print("Could not reach a required service: " + (str(e) or type(e).__name__))
            return 3
        raise


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

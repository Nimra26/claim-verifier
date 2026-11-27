"""Local web page for the Claim Verifier.

Usage: uv run python -m verifier.web            (then open http://127.0.0.1:8765)

Listens on 127.0.0.1 only. One check runs at a time (the document database is shared and the machine is small).
Each check is written to its own folder under VERIFIER_RUNS_DIR (default ~/.claim-verifier/runs) exactly as the
terminal version would read it (manifest.yaml + documents), then run with the same code; the person's questions are
answered in the page instead of the terminal.
"""
from __future__ import annotations

import verifier.appenv  # noqa: F401  - must run before any src import (see verifier/appenv.py)

import argparse
import asyncio
import datetime as dt
import json
import os
import queue
import re
import shutil
import socket
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse
from urllib.request import urlopen

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from verifier.__main__ import _Exit, _is_connection_error, _run
from verifier.config import VerifierSettings
from verifier.models import Report
from verifier.report import report_view

PAGE = Path(__file__).with_name("webui") / "index.html"
STATIC = Path(__file__).with_name("webui") / "static"  # fonts, logo, header theme: copied from Ask the Workspace
EXAMPLE = Path(__file__).with_name("testsets") / "leave-policy"
MAX_DOCS = 30
MAX_DOC_BYTES = 1_000_000
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._()-]{0,120}\.(md|txt)$", re.IGNORECASE)
ACTIVE = {"starting", "indexing", "checking", "waiting"}


class DocIn(BaseModel):
    name: str
    text: str
    date: str | None = None
    replaces: list[str] = []


class RunIn(BaseModel):
    question: str
    documents: list[DocIn]


class AnswerIn(BaseModel):
    choice: str


class _Cancelled(Exception):
    pass


@dataclass
class RunState:
    id: str
    folder: Path
    status: str = "starting"  # starting, indexing, checking, waiting, done, failed, cancelled
    message: str = ""
    question: dict | None = None
    view: dict | None = None
    started: float = field(default_factory=time.time)
    answers: queue.Queue = field(default_factory=queue.Queue)

    def public(self) -> dict:
        return {"id": self.id, "status": self.status, "message": self.message, "question": self.question,
                "report": self.view, "elapsed": round(time.time() - self.started)}


def _runs_dir() -> Path:
    return Path(os.environ.get("VERIFIER_RUNS_DIR", Path.home() / ".claim-verifier" / "runs"))


def _validate(body: RunIn) -> None:
    def bad(msg: str):
        raise HTTPException(status_code=422, detail=msg)

    if not body.question.strip():
        bad("Type a question first.")
    if not body.documents:
        bad("Add at least one document.")
    if len(body.documents) > MAX_DOCS:
        bad(f"At most {MAX_DOCS} documents per check.")
    names = [d.name for d in body.documents]
    if len(set(n.lower() for n in names)) != len(names):
        bad("Two documents have the same file name.")
    stems = [Path(n).stem.lower() for n in names]
    if len(set(stems)) != len(stems):
        bad("Two documents have the same name with a different ending (.md / .txt). Rename one.")
    for d in body.documents:
        if not SAFE_NAME.match(d.name):
            bad(f"“{d.name}”: only .md and .txt files with plain names can be checked.")
        if not d.text.strip():
            bad(f"“{d.name}” is empty.")
        if len(d.text.encode("utf-8")) > MAX_DOC_BYTES:
            bad(f"“{d.name}” is too large (limit 1 MB of text).")
        if d.date:
            try:
                dt.date.fromisoformat(d.date)
            except ValueError:
                bad(f"“{d.name}”: the date must look like 2025-03-01.")
        for r in d.replaces:
            if r not in names or r == d.name:
                bad(f"“{d.name}” replaces “{r}”, which is not another document in this check.")


def _write_folder(body: RunIn, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=False)
    docs = []
    for d in body.documents:
        (folder / d.name).write_text(d.text, encoding="utf-8")
        entry: dict = {"file": d.name}
        if d.date:
            entry["date"] = d.date
        if d.replaces:
            entry["replaces"] = list(d.replaces)
        docs.append(entry)
    manifest = {"question": body.question.strip(), "documents": docs}
    (folder / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
                                          encoding="utf-8")


def _check_tcp(url: str, default_port: int) -> bool:
    u = urlparse(url.replace("+asyncpg", ""))
    try:
        with socket.create_connection((u.hostname or "localhost", u.port or default_port), timeout=1.5):
            return True
    except OSError:
        return False


def _health(settings: VerifierSettings) -> dict:
    db = _check_tcp(settings.database_url, 5432)
    model = False
    ollama = False
    try:
        with urlopen(settings.ollama_url.rstrip("/") + "/api/tags", timeout=2) as r:  # noqa: S310 - local URL
            ollama = True
            names = [m.get("name", "") for m in json.load(r).get("models", [])]
            model = any(n.split(":")[0] == settings.embed_model.split(":")[0] for n in names)
    except Exception:  # noqa: BLE001 - any failure means "not reachable"
        pass
    if settings.backend == "api":
        claude = bool(os.environ.get("ANTHROPIC_API_KEY"))
    else:
        claude = shutil.which("claude") is not None
    return {
        "database": {"ok": db, "help": "Start the document database (see README, Setup step 1)."},
        "search": {"ok": ollama and model,
                   "help": ("Start Ollama (see README, Setup step 2)." if not ollama
                            else f"Download the search model: docker exec claim-verifier-ollama ollama pull {settings.embed_model}")},
        "claude": {"ok": claude, "help": ("Set ANTHROPIC_API_KEY." if settings.backend == "api"
                                          else "Install Claude Code and sign in (run: claude).")},
    }


def create_app(*, reasoner_factory=None, store_factory=None) -> FastAPI:
    """`reasoner_factory` / `store_factory` are for tests; by default the real Claude and database are used."""
    app = FastAPI(title="Claim Verifier", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    runs: dict[str, RunState] = {}
    lock = threading.Lock()

    def worker(state: RunState) -> None:
        def say(line: str) -> None:
            if line.startswith("Indexing"):
                state.status = "indexing"
            elif line.startswith("Checking"):
                state.status = "checking"
            elif not line.startswith("Report written"):
                state.message = line

        def ask(question: dict) -> str:
            state.question = dict(question)
            state.status = "waiting"
            choice = state.answers.get()
            if choice is None:
                raise _Cancelled()
            state.question = None
            state.status = "checking"
            return choice

        args = SimpleNamespace(folder=str(state.folder), answers=None, unattended=False)
        reasoner = reasoner_factory() if reasoner_factory else None
        store = store_factory() if store_factory else None
        try:
            code = asyncio.run(_run(args, reasoner, store, None, ask=ask, say=say))
            report = Report.model_validate_json((state.folder / "report.json").read_text(encoding="utf-8"))
            state.view = report_view(report)
            state.status = "done" if code == 0 else "failed"
            if code != 0:
                state.message = "The check could not be completed. The report lists what failed."
        except _Cancelled:
            state.status, state.message = "cancelled", "Check stopped."
        except _Exit as e:
            state.status, state.message = "failed", e.message
        except Exception as e:  # noqa: BLE001 - shown to the person in plain words
            state.status = "failed"
            state.message = ("Could not reach a required service. Check that the database and search model are running."
                             if _is_connection_error(e) else f"Something went wrong: {type(e).__name__}: {e}")
        finally:
            state.question = None

    @app.get("/", response_class=HTMLResponse)
    def page():
        return PAGE.read_text(encoding="utf-8")

    @app.get("/api/health")
    def health():
        return _health(VerifierSettings())

    @app.get("/api/example")
    def example():
        manifest = yaml.safe_load((EXAMPLE / "manifest.yaml").read_text(encoding="utf-8"))
        return {"question": manifest["question"], "documents": [
            {"name": d["file"], "text": (EXAMPLE / d["file"]).read_text(encoding="utf-8"),
             "date": str(d["date"]) if d.get("date") else None, "replaces": d.get("replaces", [])}
            for d in manifest["documents"]]}

    @app.post("/api/runs")
    def start(body: RunIn):
        _validate(body)
        with lock:
            if any(r.status in ACTIVE for r in runs.values()):
                raise HTTPException(status_code=409, detail="Another check is still running. Wait for it or stop it.")
            run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
            state = RunState(id=run_id, folder=_runs_dir() / run_id)
            _write_folder(body, state.folder)
            runs[run_id] = state
        threading.Thread(target=worker, args=(state,), daemon=True, name=f"check-{run_id}").start()
        return state.public()

    def _get(run_id: str) -> RunState:
        state = runs.get(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail="No such check.")
        return state

    @app.get("/api/runs/{run_id}")
    def status(run_id: str):
        return _get(run_id).public()

    @app.post("/api/runs/{run_id}/answer")
    def answer(run_id: str, body: AnswerIn):
        state = _get(run_id)
        q = state.question
        if state.status != "waiting" or q is None:
            raise HTTPException(status_code=409, detail="The check is not waiting for an answer.")
        if body.choice not in q["choices"]:
            raise HTTPException(status_code=422, detail="Pick one of the offered choices.")
        state.answers.put(body.choice)
        return {"ok": True}

    @app.post("/api/runs/{run_id}/cancel")
    def cancel(run_id: str):
        state = _get(run_id)
        if state.status == "waiting":
            state.answers.put(None)
            return {"ok": True}
        raise HTTPException(status_code=409, detail="A check can be stopped only while it waits for your answer.")

    @app.get("/api/runs/{run_id}/report.{ext}")
    def download(run_id: str, ext: str):
        state = _get(run_id)
        if ext not in ("md", "json") or not (state.folder / f"report.{ext}").is_file():
            raise HTTPException(status_code=404, detail="No report yet.")
        return FileResponse(state.folder / f"report.{ext}", filename=f"claim-report-{run_id}.{ext}")

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(prog="verifier.web", description="Open the Claim Verifier in your browser.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    print(f"Claim Verifier is open at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

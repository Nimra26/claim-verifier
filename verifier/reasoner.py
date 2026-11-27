"""Reasoner: the single interface to the language model.

Output models DocFacts, ExtractedClaim, ExtractedClaims live here.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import shutil
import tempfile
from typing import Protocol

from anthropic import AsyncAnthropic, transform_schema
from pydantic import BaseModel

from verifier import prompts
from verifier.models import Claim, DocumentInfo, Judgment, Passage, Verdict


class DocFacts(BaseModel):
    date: dt.date | None = None
    replaces: list[str] = []


class ExtractedClaim(BaseModel):
    text: str
    quote: str
    important: bool


class ExtractedClaims(BaseModel):
    claims: list[ExtractedClaim]
    dropped_non_factual: int = 0


# Wire schemas sent to the model as structured output. Every field is required (optional values are
# nullable, not defaulted) so they are valid for the API's strict JSON-schema output (C2).
class _DocFactsOut(BaseModel):
    date: dt.date | None
    replaces: list[str]


class _ExtractedClaimsOut(BaseModel):
    claims: list[ExtractedClaim]
    dropped_non_factual: int


_JudgmentOut = Judgment  # already all-required


class ModelOutputError(RuntimeError):
    """The model refused or returned no output matching the schema."""


class _Group(BaseModel):
    claim_id: str
    subject: str


class _Groups(BaseModel):
    groups: list[_Group]


class _Reword(BaseModel):
    query: str


class _Answer(BaseModel):
    answer: str


class Reasoner(Protocol):
    async def doc_facts(self, title: str, text: str, other_titles: dict[str, str]) -> DocFacts: ...

    async def extract_claims(self, question: str, doc: DocumentInfo, text: str) -> ExtractedClaims: ...

    async def group_claims(self, question: str, claims: list[Claim]) -> dict[str, str]: ...

    async def judge(self, claim: Claim, passages: list[Passage]) -> Judgment: ...

    async def reword(self, claim: Claim, tried: list[str]) -> str: ...

    async def write_answer(
        self, question: str, claims: list[Claim], verdicts: list[Verdict], caveat: str | None
    ) -> str: ...


def _esc(text: str) -> str:
    """Neutralise only < and > so untrusted text cannot close or open wrapper tags.

    & and quotes are left alone so the model can copy quotes word for word.
    """
    return str(text).replace("<", "&lt;").replace(">", "&gt;")


def _attr(text: str) -> str:
    """For attribute values (id=, title=, verdict=): also escape double quotes (M2). Body text keeps them."""
    return _esc(text).replace('"', "&quot;")


def _unesc(text: str) -> str:
    return text.replace("&lt;", "<").replace("&gt;", ">")


MAX_TOKENS = 16000


class ClaudeReasoner:
    """Calls Claude through the anthropic SDK with structured outputs (`output_config.format`).

    No sampling parameters and no forced tool_choice: current models (e.g. claude-sonnet-5-5) reject both.
    """

    def __init__(self, model: str, client=None):
        self.model = model
        self._client = client

    def _get_client(self):
        if self._client is None:
            self._client = AsyncAnthropic()
        return self._client

    async def _call(self, schema: type[BaseModel], system: str, user: str):
        msg = await self._get_client().messages.parse(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
        )
        if msg.stop_reason == "refusal":
            raise ModelOutputError("the model declined the request")
        if msg.parsed_output is None:
            raise ModelOutputError(f"no {schema.__name__} in the model output (stop: {msg.stop_reason})")
        return msg.parsed_output

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()

    async def doc_facts(self, title: str, text: str, other_titles: dict[str, str]) -> DocFacts:
        others = "\n".join(f'<title id="{_attr(i)}">{_esc(t)}</title>' for i, t in other_titles.items()) or "(none)"
        user = (
            f"<title>{_esc(title)}</title>\n<document>\n{_esc(text)}\n</document>\n\n"
            f"Other documents (id and title):\n{others}"
        )
        res = await self._call(_DocFactsOut, prompts.DOC_FACTS_SYSTEM, user)
        return DocFacts(date=res.date, replaces=[i for i in res.replaces if i in other_titles])

    async def extract_claims(self, question: str, doc: DocumentInfo, text: str) -> ExtractedClaims:
        user = f"Question: {_esc(question)}\n\n<document title=\"{_attr(doc.title)}\">\n{_esc(text)}\n</document>"
        res = await self._call(_ExtractedClaimsOut, prompts.EXTRACT_SYSTEM, user)
        return ExtractedClaims(
            claims=[
                ExtractedClaim(text=_unesc(c.text), quote=_unesc(c.quote), important=c.important)
                for c in res.claims
            ],
            dropped_non_factual=res.dropped_non_factual,
        )

    async def group_claims(self, question: str, claims: list[Claim]) -> dict[str, str]:
        body = "\n".join(f'<claim id="{_attr(c.id)}">{_esc(c.text)}</claim>' for c in claims)
        res = await self._call(_Groups, prompts.GROUP_SYSTEM, f"Question: {_esc(question)}\n\n{body}")
        ids = {c.id for c in claims}
        return {g.claim_id: g.subject for g in res.groups if g.claim_id in ids}

    async def judge(self, claim: Claim, passages: list[Passage]) -> Judgment:
        body = "\n".join(f'<passage id="{_attr(p.id)}">{_esc(p.text)}</passage>' for p in passages)
        user = (
            f'<claim id="{_attr(claim.id)}">{_esc(claim.text)}</claim>\n'
            f"Claim's own passage (quote):\n<passage id=\"own\">{_esc(claim.quote)}</passage>\n\n"
            f"Evidence passages:\n{body or '(none)'}"
        )
        res = Judgment.model_validate(
            (await self._call(_JudgmentOut, prompts.JUDGE_SYSTEM, user)).model_dump())
        sent = {p.id for p in passages}  # the "own" id is dropped too: stances are for evidence passages only
        return res.model_copy(update={"stances": [st for st in res.stances if st.passage_id in sent]})

    async def reword(self, claim: Claim, tried: list[str]) -> str:
        tried_txt = "\n".join(f"- {_esc(q)}" for q in tried) or "(none)"
        user = f"<claim>{_esc(claim.text)}</claim>\nQueries already tried:\n{tried_txt}"
        res = await self._call(_Reword, prompts.REWORD_SYSTEM, user)
        return res.query

    async def write_answer(
        self, question: str, claims: list[Claim], verdicts: list[Verdict], caveat: str | None
    ) -> str:
        by_id = {v.claim_id: v for v in verdicts}
        lines = []
        for c in claims:
            v = by_id.get(c.id)
            tag = f"{v.label}: {v.reason}" if v else "no verdict"
            lines.append(f'<claim id="{_attr(c.id)}" verdict="{_attr(tag)}">{_esc(c.text)}</claim>')
        user = f"Question: {_esc(question)}\n\n" + "\n".join(lines) + f"\n\nCaveat: {_esc(caveat or '(none)')}"
        res = await self._call(_Answer, prompts.ANSWER_SYSTEM, user)
        return res.answer


class ClaudeCodeReasoner(ClaudeReasoner):
    """Same prompts and schemas, sent through the local Claude Code command (`claude -p`) so the run uses the
    machine's Claude Code login instead of an API key. No tools, no plugins, no MCP servers, no settings files,
    no saved sessions: one plain structured-output turn per call."""

    def __init__(self, model: str, *, command: str = "claude", max_parallel: int = 4, timeout: float = 600.0):
        super().__init__(model)
        self.command = command
        self.timeout = timeout
        self._slots = asyncio.Semaphore(max_parallel)
        self._cwd = tempfile.mkdtemp(prefix="claim-verifier-")  # empty folder: no project files are read

    async def _call(self, schema: type[BaseModel], system: str, user: str):
        args = [
            self.command, "-p", "--model", self.model, "--output-format", "json",
            "--json-schema", json.dumps(transform_schema(schema)), "--system-prompt", system,
            "--tools", "", "--no-session-persistence", "--setting-sources", "",
            "--strict-mcp-config", "--disable-slash-commands",
        ]
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}  # use the login, not a key
        async with self._slots:
            proc = await asyncio.create_subprocess_exec(
                *args, cwd=self._cwd, env=env, stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            try:
                out, err = await asyncio.wait_for(proc.communicate(user.encode()), self.timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise ModelOutputError(f"Claude Code did not answer within {self.timeout:.0f}s")
        try:
            data = json.loads(out.decode() or "{}")
        except json.JSONDecodeError:
            data = {}
        if proc.returncode != 0 or data.get("is_error") or "structured_output" not in data:
            detail = (data.get("result") or err.decode() or f"exit {proc.returncode}").strip()[:300]
            raise ModelOutputError(f"Claude Code returned no {schema.__name__}: {detail}")
        return schema.model_validate(data["structured_output"])

    async def aclose(self) -> None:
        shutil.rmtree(self._cwd, ignore_errors=True)


class OllamaReasoner(ClaudeReasoner):
    """Same prompts and schemas, answered by a model running in the local Ollama (as Ask the Workspace does with
    qwen3.5:4b). Nothing leaves the machine. Ollama's structured output (`format` = JSON schema) keeps the replies
    in the same shape; a reply that still doesn't fit is asked for once more before it counts as a failure."""

    def __init__(self, model: str, *, url: str = "http://localhost:11434", max_parallel: int = 1,
                 timeout: float = 900.0, context: int = 16384, client=None):
        super().__init__(model)
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.context = context
        self._http = client
        self._slots = asyncio.Semaphore(max_parallel)  # one at a time: on a laptop CPU, parallel calls only queue

    def _client_http(self):
        if self._http is None:
            import httpx

            self._http = httpx.AsyncClient(timeout=self.timeout)
        return self._http

    async def _call(self, schema: type[BaseModel], system: str, user: str):
        body = {
            "model": self.model, "stream": False, "think": False,
            "format": schema.model_json_schema(),
            "options": {"temperature": 0, "num_ctx": self.context},
            "messages": [{"role": "system", "content": system + "\n\nReply with JSON only."},
                         {"role": "user", "content": user}],
        }
        last = ""
        for _ in range(2):
            async with self._slots:
                try:
                    r = await self._client_http().post(self.url + "/api/chat", json=body)
                except Exception as e:  # noqa: BLE001 - re-raised as plain words below unless it is a network error
                    if type(e).__name__ in {"ConnectError", "ConnectTimeout"}:
                        raise
                    raise ModelOutputError(f"the local model did not answer ({type(e).__name__})") from e
            if r.status_code != 200:
                raise ModelOutputError(f"the local model returned an error: {r.text[:300]}")
            last = (r.json().get("message") or {}).get("content", "")
            try:
                return schema.model_validate_json(last)
            except ValueError:
                continue
        raise ModelOutputError(f"the local model returned no valid {schema.__name__}: {last[:200]!r}")

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()


def local_model_ready(url: str, model: str) -> bool | None:
    """True when Ollama has the model, False when it doesn't, None when Ollama can't be reached."""
    from urllib.request import urlopen

    try:
        with urlopen(url.rstrip("/") + "/api/tags", timeout=3) as r:  # noqa: S310 - local URL
            names = {m.get("name", "") for m in json.load(r).get("models", [])}
    except Exception:  # noqa: BLE001
        return None
    want = model if ":" in model else model + ":latest"
    return want in names


BACKENDS = {"local": "on this computer (Ollama)", "cli": "Claude, through Claude Code", "api": "Claude, API key"}


def normalize_backend(name: str | None) -> str | None:
    """User-facing names: local | claude | api ("claude" means the Claude Code login, i.e. "cli")."""
    if name is None:
        return None
    name = name.strip().lower()
    return {"claude": "cli", "ollama": "local"}.get(name, name)


def make_reasoner(backend: str, model: str, *, ollama_url: str = "http://localhost:11434") -> ClaudeReasoner:
    """backend "local": a model in the local Ollama, nothing leaves the machine. "cli": the local Claude Code login
    (no API key). "api": ANTHROPIC_API_KEY."""
    backend = normalize_backend(backend)
    if backend == "local":
        ready = local_model_ready(ollama_url, model)
        if ready is None:
            raise RuntimeError(f"Could not reach Ollama at {ollama_url}. Start it (see README, Setup step 2).")
        if not ready:
            raise RuntimeError(f"The local model {model} is not downloaded yet. Download it with: "
                               f"docker exec claim-verifier-ollama ollama pull {model}  (or choose Claude instead).")
        return OllamaReasoner(model, url=ollama_url)
    if backend == "api":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        return ClaudeReasoner(model)
    if backend == "cli":
        if shutil.which("claude") is None:
            raise RuntimeError("The Claude Code command (claude) was not found.")
        return ClaudeCodeReasoner(model)
    raise RuntimeError(f"Unknown backend {backend!r}: use local, claude or api.")

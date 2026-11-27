"""The local backend: reasoning in Ollama, nothing sent anywhere else."""
import asyncio
import json

import httpx
import pytest

from verifier import reasoner as r
from verifier.__main__ import main
from verifier.config import VerifierSettings
from verifier.models import Claim
from verifier.tests.test_cli import make_folder, make_store


def _claim():
    return Claim(id="a#1", doc_id="a", text="Staff get 25 days.", quote="Staff get 25 days.", important=True)


def _ollama(replies, seen):
    """A fake Ollama /api/chat returning `replies` in turn."""
    it = iter(replies)

    def handle(request: httpx.Request):
        assert request.url.path == "/api/chat"
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": next(it)}})

    return httpx.AsyncClient(transport=httpx.MockTransport(handle))


def test_local_reasoner_sends_the_schema_and_reads_the_reply():
    seen = []
    m = r.OllamaReasoner("qwen3.5:4b", client=_ollama(['{"query": "annual leave days staff"}'], seen))
    assert asyncio.run(m.reword(_claim(), [])) == "annual leave days staff"
    body = seen[0]
    assert body["model"] == "qwen3.5:4b" and body["stream"] is False and body["think"] is False
    assert body["format"]["properties"]["query"] and body["options"]["temperature"] == 0


def test_local_reasoner_asks_again_once_then_gives_up():
    seen = []
    m = r.OllamaReasoner("m", client=_ollama(["not json", '{"query": "q2"}'], seen))
    assert asyncio.run(m.reword(_claim(), [])) == "q2" and len(seen) == 2
    m = r.OllamaReasoner("m", client=_ollama(["no", "still no"], []))
    with pytest.raises(r.ModelOutputError, match="no valid"):
        asyncio.run(m.reword(_claim(), []))


def test_local_reasoner_error_reply():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(500, text="model crashed")))
    with pytest.raises(r.ModelOutputError, match="model crashed"):
        asyncio.run(r.OllamaReasoner("m", client=client).reword(_claim(), []))


def test_make_reasoner_local(monkeypatch):
    monkeypatch.setattr(r, "local_model_ready", lambda url, model: True)
    assert isinstance(r.make_reasoner("local", "qwen3.5:4b"), r.OllamaReasoner)
    monkeypatch.setattr(r, "local_model_ready", lambda url, model: False)
    with pytest.raises(RuntimeError, match="ollama pull qwen3.5:4b"):
        r.make_reasoner("local", "qwen3.5:4b")
    monkeypatch.setattr(r, "local_model_ready", lambda url, model: None)
    with pytest.raises(RuntimeError, match="Could not reach Ollama"):
        r.make_reasoner("local", "qwen3.5:4b")


def test_backend_names_and_models(monkeypatch):
    assert r.normalize_backend("Claude") == "cli" and r.normalize_backend("local") == "local"
    monkeypatch.delenv("VERIFIER_BACKEND", raising=False)
    s = VerifierSettings()
    assert s.backend == "local"  # private by default
    assert s.model_for("local") == "qwen3.5:4b" and s.model_for("claude") == s.model


def test_cli_backend_flag_picks_local_and_says_when_model_is_missing(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("VERIFIER_BACKEND", "cli")
    monkeypatch.setattr(r, "local_model_ready", lambda url, model: False)
    code = main(["run", str(make_folder(tmp_path)), "--backend", "local"], store=make_store())
    out = capsys.readouterr().out
    assert code == 2 and "not downloaded yet" in out and "choose Claude" in out

import asyncio
import json

import pytest

from verifier import reasoner as r
from verifier.models import Claim, Passage


def _claim():
    return Claim(id="a#1", doc_id="a", text="Staff get 25 days.", quote="Staff get 25 days.", important=True)


class _Proc:
    def __init__(self, out: dict | str, code: int = 0):
        self._out = out if isinstance(out, str) else json.dumps(out)
        self.returncode = code
        self.stdin_seen = None

    async def communicate(self, data):
        self.stdin_seen = data
        return self._out.encode(), b""


def _patch(monkeypatch, proc):
    calls = []

    async def fake_exec(*args, **kw):
        calls.append((args, kw))
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    return calls


def test_cli_reasoner_sends_schema_and_prompt_and_drops_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "bad")
    proc = _Proc({"is_error": False, "structured_output": {"query": "reworded query"}})
    calls = _patch(monkeypatch, proc)
    rs = r.ClaudeCodeReasoner("claude-sonnet-5-5")
    out = asyncio.run(rs.reword(_claim(), []))
    asyncio.run(rs.aclose())
    assert out == "reworded query"
    args, kw = calls[0]
    assert args[:2] == ("claude", "-p")
    assert args[args.index("--model") + 1] == "claude-sonnet-5-5"
    assert args[args.index("--tools") + 1] == ""
    schema = json.loads(args[args.index("--json-schema") + 1])
    assert schema["additionalProperties"] is False
    assert "ANTHROPIC_API_KEY" not in kw["env"]
    assert b"Staff get 25 days." in proc.stdin_seen


def test_cli_reasoner_error_becomes_model_output_error(monkeypatch):
    _patch(monkeypatch, _Proc({"is_error": True, "result": "Not logged in"}, code=1))
    with pytest.raises(r.ModelOutputError, match="Not logged in"):
        asyncio.run(r.ClaudeCodeReasoner("m").judge(_claim(), [
            Passage(id="b:0", doc_id="b", source="B", text="Staff get 25 days.", score=1.0)]))


def test_cli_reasoner_garbage_output_becomes_model_output_error(monkeypatch):
    _patch(monkeypatch, _Proc("not json"))
    with pytest.raises(r.ModelOutputError):
        asyncio.run(r.ClaudeCodeReasoner("m").reword(_claim(), []))


def test_make_reasoner_picks_backend(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/claude")
    assert isinstance(r.make_reasoner("cli", "m"), r.ClaudeCodeReasoner)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        r.make_reasoner("api", "m")
    with pytest.raises(RuntimeError, match="Unknown"):
        r.make_reasoner("x", "m")

import json

import pytest

from verifier.__main__ import main
from verifier.tests.fakes import FakeEmbeddings, MemoryVectorStore
from verifier.tests.test_graph import NEW_DATE, OLD_DATE, TEXTS, fake
from verifier.config import VerifierSettings
from verifier.store import build_store


def make_folder(tmp_path):
    # "new" says it replaces "old", but "old" carries the newer date -> a person is asked.
    (tmp_path / "manifest.yaml").write_text(
        "question: How many days of annual leave does a full-time employee get?\n"
        "documents:\n"
        f"  - file: old.md\n    date: {NEW_DATE}\n"
        f"  - file: new.md\n    date: {OLD_DATE}\n"
        "  - file: faq.md\n",
        encoding="utf-8",
    )
    for k, v in TEXTS.items():
        (tmp_path / f"{k}.md").write_text(v, encoding="utf-8")
    return tmp_path


def make_store():
    return build_store(VerifierSettings(), embedding_provider=FakeEmbeddings(), vector_store=MemoryVectorStore())


def no_input(prompt=""):
    raise AssertionError("input_fn must not be called")


def test_cli_uses_answers_file(tmp_path, capsys):
    folder = make_folder(tmp_path)
    answers = tmp_path / "answers.yaml"
    answers.write_text('"replaces:new|old": "Leave Policy 2025"\n', encoding="utf-8")
    code = main(["run", str(folder), "--answers", str(answers)],
                reasoner=fake(), store=make_store(), input_fn=no_input)
    assert code == 0
    assert (folder / "report.md").is_file() and (folder / "report.json").is_file()
    out = capsys.readouterr().out
    assert "report.md" in out and "report.json" in out
    data = json.loads((folder / "report.json").read_text())
    assert data["person_answers"] == {"replaces:new|old": "Leave Policy 2025"}


def test_cli_prompts_when_no_answer(tmp_path, capsys):
    folder = make_folder(tmp_path)
    asked = []

    def input_fn(prompt=""):
        asked.append(prompt)
        return "1"

    code = main(["run", str(folder)], reasoner=fake(), store=make_store(), input_fn=input_fn)
    assert code == 0
    out = capsys.readouterr().out
    assert "1. " in out and "2. " in out
    data = json.loads((folder / "report.json").read_text())
    (qid, ans), = data["person_answers"].items()
    assert qid == "replaces:new|old" and ans == data["asked"][0]["choices"][0]
    assert len(asked) == 1


def test_cli_invalid_answers_file_entry_falls_back_to_asking(tmp_path):
    folder = make_folder(tmp_path)
    answers = tmp_path / "answers.yaml"
    answers.write_text('"replaces:new|old": "bogus"\n', encoding="utf-8")
    calls = []
    code = main(["run", str(folder), "--answers", str(answers)], reasoner=fake(), store=make_store(),
                input_fn=lambda p="": calls.append(p) or "2")
    assert code == 0 and len(calls) == 1


def test_cli_reasks_then_defaults_to_last_choice(tmp_path, capsys):
    folder = make_folder(tmp_path)
    calls = []
    code = main(["run", str(folder)], reasoner=fake(), store=make_store(),
                input_fn=lambda p="": calls.append(p) or "zzz")
    assert code == 0 and len(calls) == 3
    out = capsys.readouterr().out
    assert "neither" in out
    data = json.loads((folder / "report.json").read_text())
    assert list(data["person_answers"].values()) == ["neither"]


def test_cli_missing_folder(tmp_path, capsys):
    code = main(["run", str(tmp_path / "nope")], reasoner=fake(), store=make_store())
    assert code == 2
    assert "Traceback" not in capsys.readouterr().out


def test_cli_missing_manifest(tmp_path, capsys):
    code = main(["run", str(tmp_path)], reasoner=fake(), store=make_store())
    assert code == 2


def test_cli_missing_api_key(tmp_path, capsys, monkeypatch):
    folder = make_folder(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("VERIFIER_BACKEND", "api")
    code = main(["run", str(folder)], store=make_store())
    assert code == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().out


def test_cli_exits_4_when_the_check_could_not_run(tmp_path, capsys):
    # I1
    folder = make_folder(tmp_path)

    def extract(question, doc, text):
        raise RuntimeError("model rejected the request")

    code = main(["run", str(folder)], reasoner=fake(extract_claims=extract), store=make_store(), input_fn=no_input)
    assert code == 4
    out = capsys.readouterr().out
    assert "could not be completed" in out
    data = json.loads((folder / "report.json").read_text())
    assert data["completed"] is False


# --- M1: only network errors are "service unreachable" ---

def test_connection_error_classification():
    import socket

    from verifier.__main__ import _is_connection_error

    assert _is_connection_error(ConnectionRefusedError())
    assert _is_connection_error(socket.gaierror())
    assert _is_connection_error(TimeoutError())
    assert not _is_connection_error(PermissionError("denied"))
    assert not _is_connection_error(FileNotFoundError("x"))


def test_cli_report_write_permission_error_is_exit_2(tmp_path, capsys, monkeypatch):
    import verifier.__main__ as cli

    folder = make_folder(tmp_path)

    def deny(report, folder):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(cli, "write_report", deny)
    code = main(["run", str(folder), "--unattended"], reasoner=fake(), store=make_store(), input_fn=no_input)
    assert code == 2
    assert f"Could not write the report to {folder}" in capsys.readouterr().out


# --- M5: --unattended ---

def test_cli_unattended_answers_with_last_choice_and_lists_it(tmp_path, capsys):
    folder = make_folder(tmp_path)
    code = main(["run", str(folder), "--unattended"], reasoner=fake(), store=make_store(), input_fn=no_input)
    assert code == 0
    out = capsys.readouterr().out
    assert "replaces:new|old" in out and "neither" in out and "unattended" in out.lower()
    data = json.loads((folder / "report.json").read_text())
    assert data["person_answers"] == {"replaces:new|old": "neither"}


def test_cli_unattended_still_uses_answers_file(tmp_path):
    folder = make_folder(tmp_path)
    answers = tmp_path / "answers.yaml"
    answers.write_text('"replaces:new|old": "Leave Policy 2025"\n', encoding="utf-8")
    code = main(["run", str(folder), "--answers", str(answers), "--unattended"],
                reasoner=fake(), store=make_store(), input_fn=no_input)
    assert code == 0
    data = json.loads((folder / "report.json").read_text())
    assert data["person_answers"] == {"replaces:new|old": "Leave Policy 2025"}


# --- M6: the embedding client is closed at the end of the run ---

def test_cli_closes_embedding_client(tmp_path):
    closed = []

    class _Client:
        async def close(self):
            closed.append(True)

    emb = FakeEmbeddings()
    emb._client = _Client()
    store = build_store(VerifierSettings(), embedding_provider=emb, vector_store=MemoryVectorStore())
    code = main(["run", str(make_folder(tmp_path)), "--unattended"], reasoner=fake(), store=store,
                input_fn=no_input)
    assert code == 0 and closed == [True]


def test_cli_missing_claude_command(tmp_path, capsys, monkeypatch):
    folder = make_folder(tmp_path)
    monkeypatch.setenv("VERIFIER_BACKEND", "cli")
    monkeypatch.setattr("shutil.which", lambda name: None)
    code = main(["run", str(folder)], store=make_store())
    assert code == 2
    assert "Claude Code command" in capsys.readouterr().out

import time

import pytest
from fastapi.testclient import TestClient

from verifier import web
from verifier.tests.test_cli import make_store
from verifier.tests.test_graph import NEW_DATE, OLD_DATE, TEXTS, fake

QUESTION = "How many days of annual leave does a full-time employee get?"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("VERIFIER_RUNS_DIR", str(tmp_path / "runs"))
    return TestClient(web.create_app(reasoner_factory=fake, store_factory=make_store))


def body(**over):
    # "new" says it replaces "old", but "old" carries the newer date -> a person is asked.
    b = {"question": QUESTION, "documents": [
        {"name": "old.md", "text": TEXTS["old"], "date": str(NEW_DATE)},
        {"name": "new.md", "text": TEXTS["new"], "date": str(OLD_DATE)},
        {"name": "faq.md", "text": TEXTS["faq"]},
    ]}
    b.update(over)
    return b


def wait_for(client, run_id, statuses, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        s = client.get(f"/api/runs/{run_id}").json()
        if s["status"] in statuses:
            return s
        time.sleep(0.05)
    raise AssertionError(f"run never reached {statuses}: {s}")


def test_page_and_example(client):
    assert "Claim Verifier" in client.get("/").text
    for path in ("/static/css/app_theme.css", "/static/fonts/fonts.css", "/static/img/taleemabad-wordmark.png",
                 "/static/fonts/poppins-400-latin.woff2", "/static/favicon.svg"):
        assert client.get(path).status_code == 200, path
    ex = client.get("/api/example").json()
    assert ex["question"] and len(ex["documents"]) == 5
    assert all(d["text"].strip() for d in ex["documents"])


def test_full_check_asks_in_the_page_and_shows_the_report(client, tmp_path):
    r = client.post("/api/runs", json=body())
    assert r.status_code == 200
    run_id = r.json()["id"]

    s = wait_for(client, run_id, {"waiting", "done", "failed"})
    assert s["status"] == "waiting"
    q = s["question"]
    assert "replaces" in q["text"] and "Leave Policy 2025" in q["choices"]

    assert client.post(f"/api/runs/{run_id}/answer", json={"choice": "not a choice"}).status_code == 422
    assert client.post(f"/api/runs/{run_id}/answer", json={"choice": "Leave Policy 2025"}).status_code == 200

    s = wait_for(client, run_id, {"done", "failed"})
    assert s["status"] == "done", s
    rep = s["report"]
    labels = {c["text"]: c["label"] for c in rep["claims"]}
    assert labels["Full-time employees get 20 days of annual leave."] == "outdated"
    assert rep["answer"]["status"] == "answered"
    assert rep["settled"] and rep["settled"][0]["answer"] == "Leave Policy 2025"
    assert any(c["label"] == "outdated" for c in rep["contradictions"])

    md = client.get(f"/api/runs/{run_id}/report.md")
    assert md.status_code == 200 and "## Answer" in md.text
    manifest = (tmp_path / "runs" / run_id / "manifest.yaml").read_text()
    assert "old.md" in manifest and str(NEW_DATE) in manifest


def test_only_one_check_at_a_time_and_stop(client):
    run_id = client.post("/api/runs", json=body()).json()["id"]
    wait_for(client, run_id, {"waiting"})
    assert client.post("/api/runs", json=body()).status_code == 409
    assert client.post(f"/api/runs/{run_id}/cancel").status_code == 200
    assert wait_for(client, run_id, {"cancelled", "failed"})["status"] == "cancelled"
    assert client.post("/api/runs", json=body()).status_code == 200


@pytest.mark.parametrize("change, words", [
    ({"question": "  "}, "question"),
    ({"documents": []}, "at least one"),
    ({"documents": [{"name": "../evil.md", "text": "x"}]}, "only .md and .txt"),
    ({"documents": [{"name": "a.pdf", "text": "x"}]}, "only .md and .txt"),
    ({"documents": [{"name": "a.md", "text": "   "}]}, "empty"),
    ({"documents": [{"name": "a.md", "text": "x", "date": "1 March"}]}, "2025-03-01"),
    ({"documents": [{"name": "a.md", "text": "x"}, {"name": "a.txt", "text": "y"}]}, "same name"),
    ({"documents": [{"name": "a.md", "text": "x", "replaces": ["b.md"]}]}, "not another document"),
])
def test_bad_input_is_refused_in_plain_words(client, change, words):
    r = client.post("/api/runs", json=body(**change))
    assert r.status_code == 422
    assert words in r.json()["detail"]


def test_unknown_run_and_answer_when_not_waiting(client):
    assert client.get("/api/runs/nope").status_code == 404
    run_id = client.post("/api/runs", json=body(documents=[{"name": "faq.md", "text": TEXTS["faq"]}])).json()["id"]
    wait_for(client, run_id, {"done", "failed"})
    assert client.post(f"/api/runs/{run_id}/answer", json={"choice": "x"}).status_code == 409

import asyncio
import datetime as dt
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from verifier.config import VerifierSettings
from verifier.graph import build_graph
from verifier.models import DocumentInfo, Judgment, Stance
from verifier.reasoner import DocFacts, ExtractedClaim, ExtractedClaims
from verifier.search import EvidenceSearch
from verifier.store import build_store, index_documents
from verifier.tests.fakes import FakeEmbeddings, FakeReasoner, MemoryVectorStore

QUESTION = "How many days of annual leave does a full-time employee get?"
TEXTS = {
    "old": "# Leave Policy 2023\n\nFull-time employees get 20 days of annual leave per year.",
    "new": "# Leave Policy 2025\n\nFull-time employees get 25 days of annual leave per year.",
    "faq": "# HR FAQ\n\nLeave requests are submitted through the HR portal.",
}
TITLES = {"old": "Leave Policy 2023", "new": "Leave Policy 2025", "faq": "HR FAQ"}
OLD_DATE = dt.date(2023, 1, 15)
NEW_DATE = dt.date(2025, 3, 1)

CLAIMS = {
    "old": ExtractedClaims(claims=[ExtractedClaim(
        text="Full-time employees get 20 days of annual leave.",
        quote="Full-time employees get 20 days of annual leave per year.", important=True)]),
    "new": ExtractedClaims(claims=[ExtractedClaim(
        text="Full-time employees get 25 days of annual leave.",
        quote="Full-time employees get 25 days of annual leave per year.", important=True)],
        dropped_non_factual=1),
    "faq": ExtractedClaims(claims=[ExtractedClaim(
        text="Leave requests go through the HR portal.",
        quote="Leave requests are submitted through the HR portal.", important=False)]),
}


def leave_judge(claim, passages):
    """old and new contradict each other; everything else is unrelated."""
    rival = {"old#1": "new", "new#1": "old"}.get(claim.id)
    return Judgment(
        stances=[Stance(passage_id=p.id, stance="contradicts" if p.doc_id == rival else "unrelated")
                 for p in passages],
        states_claim=True, confidence="high", reason="scripted",
    )


def documents(old_date=OLD_DATE, new_date=NEW_DATE, new_replaces=(), faq_date=None):
    def d(i, date, replaces=()):
        return DocumentInfo(id=i, file=f"{i}.md", title=TITLES[i], date=date,
                            date_source="manifest" if date else None, replaces=list(replaces))
    return {"old": d("old", old_date), "new": d("new", new_date, new_replaces), "faq": d("faq", faq_date)}


def run(reasoner, docs, resumes=()):
    """Run the graph; feed each value in `resumes` to an interrupt in turn. Returns (results, graph, config)."""
    async def go():
        store = build_store(VerifierSettings(), embedding_provider=FakeEmbeddings(), vector_store=MemoryVectorStore())
        await index_documents(store, TEXTS, TITLES)
        search = EvidenceSearch(store.retrieval, list(TEXTS), 8)
        graph = build_graph(reasoner, search, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        results = [await graph.ainvoke({"question": QUESTION, "documents": docs, "texts": TEXTS}, config)]
        for value in resumes:
            results.append(await graph.ainvoke(Command(resume=value), config))
        return results

    return asyncio.run(go())


def fake(**kw):
    kw.setdefault("extract_claims", CLAIMS)
    kw.setdefault("judge", leave_judge)
    kw.setdefault("doc_facts", {"Leave Policy 2025": DocFacts(replaces=["old"])})
    kw.setdefault("group_claims", {"old#1": "annual leave days", "new#1": "annual leave days"})
    return FakeReasoner(**kw)


def verdicts(report):
    return {v.claim_id: v for v in report.verdicts}


def test_every_claim_gets_a_verdict():
    (out,) = run(fake(), documents())
    report = out["report"]
    assert len(report.claims) == 3
    assert len(report.verdicts) == len(report.claims)
    assert {c.id for c in report.claims} == {"old#1", "new#1", "faq#1"}
    assert report.dropped_non_factual == 1
    assert {c.id: c.subject for c in report.claims}["new#1"] == "annual leave days"
    check = next(c for c in report.checks if c.claim_id == "new#1")
    assert check.queries == ["Full-time employees get 25 days of annual leave.", "annual leave days"]


def test_recheck_stops_after_two_extra_tries():
    low = Judgment(stances=[], states_claim=True, confidence="low", reason="unsure")
    extract = {k: ExtractedClaims(claims=[c.model_copy(update={"important": False}) for c in v.claims])
               for k, v in CLAIMS.items()}
    r = fake(judge=lambda c, p: low, extract_claims=extract)
    (out,) = run(r, documents())
    checks = {c.claim_id: c for c in out["report"].checks}
    assert all(c.tries == 3 for c in checks.values())
    rewords = [a for name, a in r.calls if name == "reword" and a[0].id == "new#1"]
    assert len(rewords) == 2
    assert len(checks["new#1"].queries) == 4  # claim text, subject, two rewordings


def test_failed_judge_does_not_drop_claim():
    def judge(claim, passages):
        if claim.id == "faq#1":
            raise RuntimeError("model down")
        return leave_judge(claim, passages)

    (out,) = run(fake(judge=judge), documents())
    v = verdicts(out["report"])
    assert "faq#1" in v
    check = next(c for c in out["report"].checks if c.claim_id == "faq#1")
    assert "check failed: RuntimeError" in check.judgment.reason
    assert check.tries == 1
    assert v["faq#1"].label == "no_evidence"
    assert v["faq#1"].reason == "could not be checked (RuntimeError)"


def test_interrupt_and_resume():
    # "new" says it replaces "old", but "old" carries the newer date -> ask a person.
    docs = documents(old_date=NEW_DATE, new_date=OLD_DATE)
    first, second = run(fake(), docs, resumes=["Leave Policy 2025"])
    assert "__interrupt__" in first
    asked = first["__interrupt__"][0].value
    assert asked["kind"] == "replaces_vs_date" and asked["id"] == "replaces:new|old"
    report = second["report"]
    assert report.person_answers == {"replaces:new|old": "Leave Policy 2025"}
    assert [q.id for q in report.asked] == ["replaces:new|old"]
    v = verdicts(report)
    assert v["old#1"].label == "outdated" and v["old#1"].settled_by_person is True
    assert v["new#1"].label == "single_source" and v["new#1"].settled_by_person is True
    assert report.answer.status == "answered" and report.answer.claim_ids == ["new#1"]


def test_interrupt_neither_gives_not_enough_evidence():
    docs = documents(old_date=NEW_DATE, new_date=OLD_DATE)
    _, second = run(fake(), docs, resumes=["neither"])
    report = second["report"]
    v = verdicts(report)
    assert v["old#1"].label == "contradicted" and v["old#1"].settled_by_person is False
    assert report.answer.status == "not_enough_evidence"


def test_end_to_end_answer():
    (out,) = run(fake(), documents())
    report = out["report"]
    assert report.answer.status == "answered"
    assert report.answer.claim_ids == ["new#1"]
    assert "Leave Policy 2023" in report.answer.caveat and "replaced" in report.answer.caveat
    v = verdicts(report)
    assert v["old#1"].label == "outdated"
    assert v["new#1"].label == "single_source"
    assert report.person_answers == {}


def test_doc_facts_merge_manifest_date_wins_and_replaces_union():
    r = fake(doc_facts={
        "Leave Policy 2025": DocFacts(date=dt.date(2020, 1, 1), replaces=["old"]),
        "HR FAQ": DocFacts(date=dt.date(2024, 6, 10)),
    })
    docs = documents(new_replaces=["faq"])
    first, second = run(r, docs, resumes=["yes"])
    asked = first["__interrupt__"][0].value
    assert asked["kind"] == "guessed_date" and asked["doc_id"] == "faq"
    by_id = {d.id: d for d in second["report"].documents}
    assert by_id["new"].date == NEW_DATE and by_id["new"].date_source == "manifest"
    assert sorted(by_id["new"].replaces) == ["faq", "old"]
    assert by_id["faq"].date == dt.date(2024, 6, 10) and by_id["faq"].date_source == "text"


def test_guessed_date_no_clears_date():
    r = fake(doc_facts={"Leave Policy 2025": DocFacts(replaces=["old"]), "HR FAQ": DocFacts(date=dt.date(2024, 6, 10))})
    first, second = run(r, documents(), resumes=["no"])
    assert first["__interrupt__"][0].value["kind"] == "guessed_date"
    by_id = {d.id: d for d in second["report"].documents}
    assert by_id["faq"].date is None and by_id["faq"].date_source is None
    assert second["report"].person_answers == {"date:faq": "no"}


def test_low_confidence_false_marks_contradicted():
    def judge(claim, passages):
        j = leave_judge(claim, passages)
        return j.model_copy(update={"confidence": "low"}) if claim.id == "new#1" else j

    first, second = run(fake(judge=judge), documents(), resumes=["false"])
    asked = first["__interrupt__"][0].value
    assert asked["kind"] == "low_confidence" and asked["claim_id"] == "new#1"
    v = verdicts(second["report"])["new#1"]
    assert v.label == "contradicted" and v.settled_by_person is True
    assert v.reason == "a person said this is false"


def test_low_confidence_true_keeps_label_settled():
    def judge(claim, passages):
        j = leave_judge(claim, passages)
        return j.model_copy(update={"confidence": "low"}) if claim.id == "new#1" else j

    _, second = run(fake(judge=judge), documents(), resumes=["true"])
    v = verdicts(second["report"])["new#1"]
    assert v.label == "single_source" and v.settled_by_person is True


def _old_fails(claim, passages):
    if claim.id == "old#1":
        raise RuntimeError("model down")
    return leave_judge(claim, passages)


def test_failed_check_cant_tell_is_not_used_for_answer():
    first, second = run(fake(judge=_old_fails), documents(), resumes=["can't tell"])
    asked = first["__interrupt__"][0].value
    assert asked["kind"] == "low_confidence" and asked["claim_id"] == "old#1"
    report = second["report"]
    v = verdicts(report)["old#1"]
    assert v.label == "no_evidence" and v.reason == "could not be checked (RuntimeError)"
    assert report.answer.status == "answered" and report.answer.claim_ids == ["new#1"]


def test_failed_check_person_says_true_is_single_source_settled():
    _, second = run(fake(judge=_old_fails), documents(), resumes=["true"])
    v = verdicts(second["report"])["old#1"]
    assert v.label == "single_source" and v.settled_by_person is True


def test_extract_failure_for_one_doc_is_recorded_and_run_continues():
    def extract(question, doc, text):
        if doc.id == "faq":
            raise ValueError("bad output")
        return CLAIMS[doc.id]

    (out,) = run(fake(extract_claims=extract), documents())
    report = out["report"]
    assert {c.id for c in report.claims} == {"old#1", "new#1"}
    assert report.failures == ["Could not read claims from HR FAQ: ValueError"]
    assert report.answer.claim_ids == ["new#1"]


def test_doc_facts_failure_keeps_manifest_facts():
    def facts(title, text, others):
        if title == "Leave Policy 2025":
            raise ValueError("bad output")
        return DocFacts()

    docs = documents(new_replaces=["old"])
    (out,) = run(fake(doc_facts=facts), docs)
    report = out["report"]
    assert report.failures == ["Could not read date or replaces from Leave Policy 2025: ValueError"]
    by_id = {d.id: d for d in report.documents}
    assert by_id["new"] == docs["new"]
    assert report.answer.claim_ids == ["new#1"]


# --- I1: model failures must not read like a normal answer ---

def test_all_extracts_failed_says_check_could_not_run():
    def extract(question, doc, text):
        raise RuntimeError("model rejected the request")

    (out,) = run(fake(extract_claims=extract), documents())
    report = out["report"]
    assert report.completed is False
    assert report.answer.status == "not_enough_evidence"
    assert report.answer.text.startswith("The check could not be completed: ")
    assert "Could not read claims from HR FAQ: RuntimeError" in report.answer.text


def test_zero_claims_with_failures_is_incomplete():
    def extract(question, doc, text):
        if doc.id == "faq":
            raise RuntimeError("x")
        return ExtractedClaims(claims=[])

    (out,) = run(fake(extract_claims=extract), documents())
    assert out["report"].completed is False
    assert out["report"].answer.text.startswith("The check could not be completed: ")


def test_zero_claims_without_failures_is_a_normal_not_enough_evidence():
    (out,) = run(fake(extract_claims=lambda q, d, t: ExtractedClaims(claims=[])), documents())
    assert out["report"].completed is True
    assert out["report"].answer.text == "Not enough evidence to answer."


# --- I5: group and write_answer failures are guarded ---

def test_group_failure_uses_claim_text_as_subject_and_records_failure():
    def group(question, claims):
        raise RuntimeError("bad output")

    (out,) = run(fake(group_claims=group), documents())
    report = out["report"]
    assert {c.id: c.subject for c in report.claims} == {c.id: c.text for c in report.claims}
    assert "Could not group claims by subject: RuntimeError" in report.failures
    assert report.completed is True and len(report.verdicts) == 3


def test_write_answer_failure_uses_template_answer():
    def write(question, claims, verdicts, caveat):
        raise RuntimeError("bad output")

    (out,) = run(fake(write_answer=write), documents())
    report = out["report"]
    assert report.answer.status == "answered" and report.answer.claim_ids == ["new#1"]
    assert "Full-time employees get 25 days of annual leave." in report.answer.text
    assert "Could not write the answer: RuntimeError" in report.failures


# --- I4: low confidence answer claims carry a caveat ---

def test_low_confidence_answer_claim_gets_caveat():
    def judge(claim, passages):
        j = leave_judge(claim, passages)
        return j.model_copy(update={"confidence": "low"}) if claim.id == "new#1" else j

    _, second = run(fake(judge=judge), documents(), resumes=["can't tell"])
    report = second["report"]
    assert report.answer.claim_ids == ["new#1"]
    assert "low confidence" in report.answer.caveat


def test_report_claims_carry_subjects():
    (out,) = run(fake(), documents())
    subjects = {c.id: c.subject for c in out["report"].claims}
    assert subjects["old#1"] == subjects["new#1"] == "annual leave days"

from verifier.models import (
    Answer, Claim, ClaimCheck, PersonQuestion, DocumentInfo, Judgment, Passage, Report, Stance, Verdict,
)
from verifier.report import render_markdown, write_report


def _check(claim_id, passages):
    return ClaimCheck(
        claim_id=claim_id, queries=["q"], passages=passages, tries=1,
        judgment=Judgment(
            stances=[Stance(passage_id=p.id, stance="supports") for p in passages],
            states_claim=True, confidence="high", reason="r",
        ),
    )


def make_report(status="answered") -> Report:
    docs = [
        DocumentInfo(id="d1", file="a.md", title="Budget 2025", date="2025-03-01"),
        DocumentInfo(id="d2", file="b.pdf", title="Budget 2024", date="2024-03-01"),
    ]
    claims = [
        Claim(id="c1", doc_id="d1", text="The budget is 5M.", quote="5M", important=True),
        Claim(id="c2", doc_id="d1", text="Staff count is 40.", quote="40", important=True),
        Claim(id="c3", doc_id="d1", text="Office has a | pipe.", quote="x", important=False),
    ]
    p_long = Passage(id="p1", doc_id="d2", source="b.pdf", text="x" * 300, page=4, score=0.9)
    p_old = Passage(id="p2", doc_id="d2", source="b.pdf", text="Budget was 4M.\nLine two | cell", page=2, score=0.8)
    p_md = Passage(id="p3", doc_id="d1", source="a.md", text="Staff is 40.", score=0.7)
    checks = [_check("c1", [p_old]), _check("c2", [p_long, p_md]), _check("c3", [])]
    verdicts = [
        Verdict(claim_id="c1", label="outdated", reason="r", evidence=["p2"], beat_doc_title="Budget 2024"),
        Verdict(claim_id="c2", label="verified", reason="r", evidence=["p1", "p3"]),
        Verdict(claim_id="c3", label="no_evidence", reason="r", evidence=[]),
    ]
    if status == "answered":
        ans = Answer(status="answered", text="The budget is 5M.", caveat="It replaced Budget 2024's figure.", claim_ids=["c1"])
    else:
        ans = Answer(status="not_enough_evidence", text="", claim_ids=[])
    return Report(
        question="What is the budget?", answer=ans, documents=docs, claims=claims,
        checks=checks, verdicts=verdicts, person_answers={}, dropped_non_factual=2,
    )


def test_answer_first():
    md = render_markdown(make_report())
    assert md.index("## Answer") < md.index("## Claims")
    assert "The budget is 5M." in md
    assert "*It replaced Budget 2024's figure.*" in md
    assert "dropped" in md.splitlines()[-1].lower() or "dropped" in md.lower()


def test_not_enough_evidence_wording():
    md = render_markdown(make_report("nope"))
    assert "Not enough evidence to answer." in md


def test_every_claim_in_table():
    md = render_markdown(make_report())
    for text in ("The budget is 5M.", "Staff count is 40.", "Office has a \\| pipe."):
        assert text in md
    assert "Only one document says this" not in md.split("## Contradictions")[0]
    assert "Outdated" in md and "Verified" in md and "No evidence" in md
    assert "Budget 2025" in md
    # quote truncated, page cited, newline/pipe escaped
    assert "x" * 200 + "…" in md and "x" * 201 not in md
    assert "Budget 2024, p. 4" in md
    assert "Line two \\| cell" in md
    for line in md.splitlines():
        if line.startswith("| ") and "Budget was 4M." in line:
            assert "\n" not in line


def test_contradiction_shows_both_sides():
    md = render_markdown(make_report())
    section = md.split("## Contradictions")[1]
    assert "The budget is 5M." in section
    assert "Budget was 4M." in section
    assert "Budget 2024" in section


def test_failures_section():
    r = make_report()
    r.failures = ["Could not read b.pdf"]
    md = render_markdown(r)
    assert "## Could not check" in md and "Could not read b.pdf" in md


def test_json_round_trips(tmp_path):
    r = make_report()
    md_path, json_path = write_report(r, tmp_path)
    assert md_path.name == "report.md" and json_path.name == "report.json"
    assert Report.model_validate_json(json_path.read_text()) == r


def test_settled_by_person_shows_question_and_answer():
    r = make_report()
    assert "## Settled by a person" not in render_markdown(r)
    r.asked = [PersonQuestion(id="q1", kind="replaces_vs_date", text="Which policy is current?", choices=["A", "B"])]
    r.person_answers = {"q1": "Policy B"}
    md = render_markdown(r)
    assert "Which policy is current? → Policy B" in md


def _table_row(md, text):
    return next(line for line in md.splitlines() if line.startswith("| ") and text in line)


def test_why_line_uses_subject_logic():
    # I2: only an outdated claim (newer doc won) is not "documents disagree".
    md = render_markdown(make_report("nope"))
    assert "The documents disagree" not in md
    assert "No checked statement in the documents answers this question." in md
    r = make_report("nope")
    r.claims[0] = r.claims[0].model_copy(update={"subject": "budget"})
    r.claims[2] = r.claims[2].model_copy(update={"subject": "budget"})
    r.verdicts[2] = Verdict(claim_id="c3", label="contradicted", reason="r", evidence=[])
    assert "The documents disagree" in render_markdown(r)


def test_single_source_evidence_shows_own_quote_and_older_side():
    # I3
    r = make_report()
    r.verdicts[0] = Verdict(claim_id="c1", label="single_source", reason="Only Budget 2025 states this.",
                            evidence=["p2"], beat_doc_title="Budget 2024")
    row = _table_row(render_markdown(r), "The budget is 5M.")
    assert "Stated in Budget 2025: “5M”" in row
    assert "Older document said: “Budget was 4M." in row


def test_single_source_without_evidence_never_empty():
    r = make_report()
    r.verdicts[0] = Verdict(claim_id="c1", label="single_source", reason="r", evidence=[], settled_by_person=True)
    row = _table_row(render_markdown(r), "The budget is 5M.")
    assert "Stated in Budget 2025: “5M”" in row


def test_table_has_reason_and_low_confidence_mark():
    # I4
    r = make_report()
    r.verdicts[1] = Verdict(claim_id="c2", label="verified", reason="Budget 2024 also states this.",
                            evidence=["p1"])
    r.checks[1] = r.checks[1].model_copy(update={"judgment": r.checks[1].judgment.model_copy(
        update={"confidence": "low"})})
    md = render_markdown(r)
    assert "| Claim | Label | Why | Source | Evidence |" in md
    row = _table_row(md, "Staff count is 40.")
    assert "Verified (low confidence)" in row and "Budget 2024 also states this." in row
    r.verdicts[1] = r.verdicts[1].model_copy(update={"settled_by_person": True})
    assert "(low confidence)" not in _table_row(render_markdown(r), "Staff count is 40.")


def test_incomplete_run_says_check_could_not_be_completed():
    # I1
    r = make_report("nope")
    r.completed = False
    r.answer = Answer(status="not_enough_evidence",
                      text="The check could not be completed: Could not read claims from Budget 2025: X",
                      claim_ids=[])
    md = render_markdown(r)
    answer = md.split("## Answer")[1].split("## Claims")[0]
    assert "The check could not be completed: Could not read claims from Budget 2025: X" in answer
    assert "No checked statement" not in answer

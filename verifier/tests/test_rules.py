import datetime as dt

from verifier.models import (
    Claim,
    ClaimCheck,
    DocumentInfo,
    Judgment,
    Passage,
    Stance,
    Verdict,
)
from verifier.rules import choose_answer, label_claim, needs_recheck, supersession

OLD = dt.date(2023, 1, 15)
NEW = dt.date(2025, 3, 1)
TEXT = "The fee is 100 dollars."


def doc(id, date=None, replaces=(), date_source="manifest"):
    return DocumentInfo(
        id=id, file=f"{id}.txt", title=f"Title {id}", date=date,
        date_source=date_source if date else None, replaces=list(replaces),
    )


def claim(doc_id="a", important=True, id="c1"):
    return Claim(id=id, doc_id=doc_id, text="fee is 100", quote="fee is 100", important=important)


def check(stances, confidence="high", states=True, claim_id="c1"):
    passages = [
        Passage(id=pid, doc_id=d, source=d, text="x", score=1.0) for pid, d, _ in stances
    ]
    return ClaimCheck(
        claim_id=claim_id, queries=["q"], passages=passages, tries=1,
        judgment=Judgment(
            stances=[Stance(passage_id=pid, stance=s) for pid, _, s in stances],
            states_claim=states, confidence=confidence, reason="r",
        ),
    )


def run(c, ch, docs, text=TEXT):
    return label_claim(c, ch, {d.id: d for d in docs}, text)


def test_newer_wins():
    v, q = run(claim("old"), check([("p1", "new", "contradicts")]), [doc("old", OLD), doc("new", NEW)])
    assert v.label == "outdated" and v.evidence == ["p1"] and q == []


def test_newer_claim_vs_older_is_single_source():
    v, _ = run(claim("new"), check([("p1", "old", "contradicts")]), [doc("old", OLD), doc("new", NEW)])
    assert v.label == "single_source"
    assert v.beat_doc_title == "Title old"


def test_replaces_wins_when_dates_agree():
    v, _ = run(claim("old"), check([("p1", "new", "contradicts")]),
               [doc("old", OLD), doc("new", NEW, replaces=["old"])])
    assert v.label == "outdated"


def test_replaces_older_dated_than_replaced_asks_person():
    docs = [doc("old", OLD, replaces=["new"]), doc("new", NEW)]
    v, q = run(claim("new"), check([("p1", "old", "contradicts")]), docs)
    assert v.label == "contradicted"
    assert len(q) == 1 and q[0].kind == "replaces_vs_date"
    assert q[0].id == "replaces:new|old"
    assert q[0].choices == ["Title new", "Title old", "neither"]


def test_undated_never_outdates():
    v, _ = run(claim("a"), check([("p1", "u", "contradicts")]), [doc("a", NEW), doc("u")])
    assert v.label == "contradicted"


def test_same_date_conflict_is_contradicted():
    v, _ = run(claim("a"), check([("p1", "b", "contradicts")]), [doc("a", NEW), doc("b", NEW)])
    assert v.label == "contradicted"


def test_support_from_other_doc_is_verified():
    v, _ = run(claim("a"), check([("p1", "b", "supports")]), [doc("a", NEW), doc("b", OLD)])
    assert v.label == "verified" and v.evidence == ["p1"]


def test_own_doc_passage_never_counts():
    v, _ = run(claim("a"), check([("p1", "a", "supports")]), [doc("a", NEW)])
    assert v.label == "single_source"


def test_quote_not_in_document_is_no_evidence():
    v, _ = run(claim("a"), check([("p1", "b", "supports")]), [doc("a"), doc("b")], text="nothing")
    assert v.label == "no_evidence"


def test_states_claim_false_is_no_evidence():
    v, _ = run(claim("a"), check([], states=False), [doc("a")])
    assert v.label == "no_evidence"


def test_low_confidence_important_asks_person():
    v, q = run(claim("a"), check([], confidence="low"), [doc("a")])
    assert [x.kind for x in q] == ["low_confidence"]
    assert q[0].id == "confidence:c1" and q[0].choices == ["true", "false", "can't tell"]


def test_low_confidence_unimportant_does_not_ask():
    _, q = run(claim("a", important=False), check([], confidence="low"), [doc("a")])
    assert q == []


def test_guessed_date_asks_once_per_doc():
    docs = [doc("a", NEW, date_source="text")]
    _, q = run(claim("a"), check([]), docs)
    assert [x.id for x in q] == ["date:a"] and q[0].kind == "guessed_date" and q[0].doc_id == "a"


def test_outdated_beats_ask_person_and_question_still_asked():
    mid = dt.date(2024, 1, 1)
    docs = [doc("a", mid), doc("n", NEW), doc("r", OLD, replaces=["a"])]
    v, q = run(claim("a"), check([("p1", "n", "contradicts"), ("p2", "r", "contradicts")]), docs)
    assert v.label == "outdated"
    assert [x.kind for x in q] == ["replaces_vs_date"]


def test_replaces_question_text_names_right_document_both_directions():
    # other doc says it replaces the claim's (newer) doc
    docs = [doc("a", NEW), doc("b", OLD, replaces=["a"])]
    _, q = run(claim("a"), check([("p1", "b", "contradicts")]), docs)
    assert q[0].text == ("Title b says it replaces Title a, but Title a has a newer date. "
                         "Which one is current?")
    # claim's own doc says it replaces a newer-dated doc
    docs = [doc("a", OLD, replaces=["b"]), doc("b", NEW)]
    _, q = run(claim("a"), check([("p1", "b", "contradicts")]), docs)
    assert q[0].text == ("Title a says it replaces Title b, but Title b has a newer date. "
                         "Which one is current?")


def test_mixed_undecided_and_current_is_contradicted():
    docs = [doc("a", NEW), doc("o", OLD), doc("u")]
    v, _ = run(claim("a"), check([("p1", "o", "contradicts"), ("p2", "u", "contradicts")]), docs)
    assert v.label == "contradicted"


def test_all_current_with_support_is_verified_and_beat_title():
    docs = [doc("a", NEW), doc("o", OLD), doc("s", NEW)]
    v, _ = run(claim("a"), check([("p1", "o", "contradicts"), ("p2", "s", "supports")]), docs)
    assert v.label == "verified" and v.beat_doc_title == "Title o"


def test_unknown_passage_stance_ignored():
    ch = check([("p1", "b", "supports")])
    ch.judgment.stances.append(Stance(passage_id="ghost", stance="contradicts"))
    v, _ = run(claim("a"), ch, [doc("a", OLD), doc("b", NEW)])
    assert v.label == "verified"


def test_needs_recheck():
    assert needs_recheck(check([], confidence="low").judgment)
    assert not needs_recheck(check([], confidence="medium").judgment)


def V(cid, label, **kw):
    return Verdict(claim_id=cid, label=label, reason="r", evidence=[], **kw)


def test_answer_prefers_verified():
    cl = [claim(id="c1"), claim(id="c2")]
    s, ids, cav = choose_answer(cl, {"c1": V("c1", "single_source"), "c2": V("c2", "verified")})
    assert s == "answered" and ids == ["c2"] and cav is None


def test_answer_single_source_has_caveat():
    s, ids, cav = choose_answer([claim()], {"c1": V("c1", "single_source", beat_doc_title="Old Doc")})
    assert s == "answered" and ids == ["c1"]
    assert "Only one document" in cav and "It replaced Old Doc's figure." in cav


def test_answer_unsettled_contradiction_declines():
    cl = [claim(id="c1"), claim(id="c2")]
    s, ids, _ = choose_answer(cl, {"c1": V("c1", "contradicted"), "c2": V("c2", "single_source")})
    assert s == "not_enough_evidence" and ids == []


def test_answer_settled_contradiction_does_not_block():
    cl = [claim(id="c1"), claim(id="c2")]
    vs = {"c1": V("c1", "contradicted", settled_by_person=True), "c2": V("c2", "single_source")}
    s, ids, _ = choose_answer(cl, vs)
    assert s == "answered" and ids == ["c2"]


def test_answer_settled_contradiction_alone_is_not_usable():
    s, _, _ = choose_answer([claim()], {"c1": V("c1", "contradicted", settled_by_person=True)})
    assert s == "not_enough_evidence"


def test_answer_none_when_no_important_claim():
    s, ids, cav = choose_answer([claim(important=False)], {"c1": V("c1", "verified")})
    assert s == "not_enough_evidence" and ids == [] and cav is None


def test_supersession_table():
    assert supersession(doc("a", NEW), doc("b", OLD, replaces=["a"])) == "ask_person"
    assert supersession(doc("a", OLD), doc("b", NEW, replaces=["a"])) == "claim_outdated"
    assert supersession(doc("a", NEW, replaces=["b"]), doc("b", OLD)) == "claim_current"
    assert supersession(doc("a", OLD, replaces=["b"]), doc("b", NEW)) == "ask_person"


# --- person answers on a replaces-vs-date clash (Ruling 7: pair_winner) ---

CLASH = [doc("old", NEW), doc("new", OLD, replaces=["old"])]  # "new" replaces "old" but "old" is newer-dated


def test_pair_winner_other_doc_makes_claim_outdated_and_settled():
    v, q = label_claim(claim("old"), check([("p1", "new", "contradicts")]), {d.id: d for d in CLASH}, TEXT,
                       pair_winner={"replaces:new|old": "new"})
    assert v.label == "outdated" and v.settled_by_person is True and v.evidence == ["p1"]
    assert [x for x in q if x.kind == "replaces_vs_date"] == []


def test_pair_winner_claim_doc_makes_claim_current_and_settled():
    v, q = label_claim(claim("new"), check([("p1", "old", "contradicts")]), {d.id: d for d in CLASH}, TEXT,
                       pair_winner={"replaces:new|old": "new"})
    assert v.label == "single_source" and v.settled_by_person is True
    assert v.beat_doc_title == "Title old"
    assert [x for x in q if x.kind == "replaces_vs_date"] == []


def test_pair_winner_neither_stays_contradicted_unsettled_no_question():
    v, q = label_claim(claim("old"), check([("p1", "new", "contradicts")]), {d.id: d for d in CLASH}, TEXT,
                       pair_winner={"replaces:new|old": "neither"})
    assert v.label == "contradicted" and v.settled_by_person is False
    assert [x for x in q if x.kind == "replaces_vs_date"] == []


def test_pair_winner_for_other_pair_does_not_change_label():
    v, q = label_claim(claim("old"), check([("p1", "new", "contradicts")]), {d.id: d for d in CLASH}, TEXT,
                       pair_winner={"replaces:x|y": "x"})
    assert v.label == "contradicted" and v.settled_by_person is False
    assert [x.kind for x in q] == ["replaces_vs_date"]


# --- I2: the answer is chosen per subject group ---

def sc(id, subject, important=True, text=None):
    return Claim(id=id, doc_id="a", text=text or f"text {id}", quote="q", important=important, subject=subject)


def test_answer_contradiction_in_other_subject_does_not_block():
    cl = [sc("c1", "annual leave days"), sc("c2", "sick leave days")]
    s, ids, cav = choose_answer(cl, {"c1": V("c1", "single_source"), "c2": V("c2", "contradicted")})
    assert s == "answered" and ids == ["c1"]
    assert "disagree about sick leave days" in cav


def test_answer_contradiction_in_same_subject_blocks_even_if_not_important():
    cl = [sc("c1", "annual leave days"), sc("c2", "annual leave days", important=False)]
    s, ids, _ = choose_answer(cl, {"c1": V("c1", "single_source"), "c2": V("c2", "contradicted")})
    assert s == "not_enough_evidence" and ids == []


def test_answer_contradiction_in_unrelated_subject_is_ignored():
    cl = [sc("c1", "annual leave days"), sc("c2", "office hours", important=False)]
    s, ids, cav = choose_answer(cl, {"c1": V("c1", "single_source"), "c2": V("c2", "contradicted")})
    assert s == "answered" and ids == ["c1"] and cav == "Only one document states this."


def test_verified_beats_single_source_only_within_subject():
    cl = [sc("c1", "annual leave days"), sc("c2", "sick leave days"), sc("c3", "annual leave days")]
    vs = {"c1": V("c1", "verified"), "c2": V("c2", "single_source"), "c3": V("c3", "single_source")}
    s, ids, cav = choose_answer(cl, vs)
    assert s == "answered" and ids == ["c1", "c2"]
    assert cav == "Only one document states this about sick leave days."


def test_answer_blockers_only_in_answer_subjects():
    from verifier.rules import answer_blockers

    cl = [sc("c1", "annual leave days"), sc("c2", "annual leave days", important=False),
          sc("c3", "office hours", important=False)]
    vs = {"c1": V("c1", "outdated"), "c2": V("c2", "contradicted"), "c3": V("c3", "contradicted")}
    assert answer_blockers(cl, vs) == ["c2"]
    vs["c2"] = V("c2", "contradicted", settled_by_person=True)
    assert answer_blockers(cl, vs) == []

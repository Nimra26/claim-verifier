"""Labelling rules. Pure Python: no I/O, no model calls."""

from typing import Literal

from verifier.models import Claim, ClaimCheck, DocumentInfo, Judgment, PersonQuestion, Verdict
from verifier.text import compare_dates, quote_in_document

Supersession = Literal["claim_outdated", "claim_current", "undecided", "ask_person"]


def supersession(claim_doc: DocumentInfo, other_doc: DocumentInfo) -> Supersession:
    by_date = compare_dates(claim_doc.date, other_doc.date)  # "a_newer" = claim doc newer
    if claim_doc.id in other_doc.replaces:
        return "ask_person" if by_date == "a_newer" else "claim_outdated"
    if other_doc.id in claim_doc.replaces:
        return "ask_person" if by_date == "b_newer" else "claim_current"
    if by_date in ("unknown", "same"):
        return "undecided"
    return "claim_outdated" if by_date == "b_newer" else "claim_current"


def needs_recheck(judgment: Judgment) -> bool:
    return judgment.confidence == "low"


def label_claim(
    claim: Claim,
    check: ClaimCheck,
    docs: dict[str, DocumentInfo],
    doc_text: str,
    *,
    pair_winner: dict[str, str] | None = None,
) -> tuple[Verdict, list[PersonQuestion]]:
    """Label one claim.

    `pair_winner` holds a person's answers to replaces-vs-date questions: question id
    ("replaces:{a}|{b}") -> winning doc id, or "neither". An answered pair is decided by that
    answer (no question is emitted for it); a label decided by it is `settled_by_person`.
    """
    pair_winner = pair_winner or {}
    claim_doc = docs[claim.doc_id]
    judgment = check.judgment
    questions: list[PersonQuestion] = []

    # Stances on passages from other documents, matched by passage id.
    by_id = {p.id: p for p in check.passages}
    supports: list[str] = []
    contradicts: dict[str, list[str]] = {}  # other doc id -> passage ids
    for st in judgment.stances:
        p = by_id.get(st.passage_id)
        if p is None or p.doc_id == claim.doc_id or p.doc_id not in docs:
            continue
        if st.stance == "supports":
            supports.append(p.id)
        elif st.stance == "contradicts":
            contradicts.setdefault(p.doc_id, []).append(p.id)

    by_person: set[str] = set()  # other doc ids whose pair outcome a person decided

    def verdict(label, reason, evidence, beat=None, settled=False) -> Verdict:
        return Verdict(claim_id=claim.id, label=label, reason=reason, evidence=evidence,
                       beat_doc_title=beat, settled_by_person=settled)

    def pair_outcome(oid: str) -> Supersession:
        a, b = sorted((claim_doc.id, oid))
        winner = pair_winner.get(f"replaces:{a}|{b}")
        if winner is None:
            return supersession(claim_doc, docs[oid])
        by_person.add(oid)
        if winner == claim_doc.id:
            return "claim_current"
        if winner == oid:
            return "claim_outdated"
        return "undecided"  # "neither" (or an unknown answer)

    result: Verdict | None = None
    if not quote_in_document(claim.quote, doc_text) or judgment.states_claim is False:
        result = verdict(
            "no_evidence",
            f"{claim_doc.title} does not actually state this, so there is nothing to check.",
            [],
        )
    elif contradicts:
        outcomes = {oid: pair_outcome(oid) for oid in sorted(contradicts)}
        for oid, out in outcomes.items():
            if out == "ask_person":
                a, b = sorted((claim_doc.id, oid))
                if claim_doc.id in docs[oid].replaces:
                    replacer, replaced = docs[oid], claim_doc
                else:
                    replacer, replaced = claim_doc, docs[oid]
                questions.append(PersonQuestion(
                    id=f"replaces:{a}|{b}", kind="replaces_vs_date", claim_id=claim.id,
                    text=(f"{replacer.title} says it replaces {replaced.title}, but "
                          f"{replaced.title} has a newer date. Which one is current?"),
                    choices=[claim_doc.title, docs[oid].title, "neither"],
                ))
        outdated = [o for o, out in outcomes.items() if out == "claim_outdated"]
        unsettled = [o for o, out in outcomes.items() if out in ("ask_person", "undecided")]
        if outdated:
            titles = ", ".join(docs[o].title for o in outdated)
            result = verdict(
                "outdated",
                f"{titles} is newer or replaces {claim_doc.title} and says something different.",
                [pid for o in outdated for pid in contradicts[o]],
                settled=any(o in by_person for o in outdated),
            )
        elif unsettled:
            titles = ", ".join(docs[o].title for o in unsettled)
            result = verdict(
                "contradicted",
                f"{titles} says something different and we cannot tell which is current.",
                [pid for o in unsettled for pid in contradicts[o]],
            )
    if result is None:
        beaten = sorted(contradicts)  # all claim_current here (or no contradictions)
        beat = docs[beaten[0]].title if beaten else None
        settled = any(o in by_person for o in beaten)
        if supports:
            titles = ", ".join(sorted({docs[by_id[p].doc_id].title for p in supports}))
            result = verdict("verified", f"{titles} also states this.", supports, beat, settled)
        else:
            result = verdict(
                "single_source",
                f"Only {claim_doc.title} states this.",
                [pid for o in beaten for pid in contradicts[o]],
                beat,
                settled,
            )

    if judgment.confidence == "low" and claim.important:
        questions.append(PersonQuestion(
            id=f"confidence:{claim.id}", kind="low_confidence", claim_id=claim.id,
            text=f"We are not sure about this: {claim.text}. Is it true?",
            choices=["true", "false", "can't tell"],
        ))
    if claim_doc.date_source == "text":
        questions.append(PersonQuestion(
            id=f"date:{claim_doc.id}", kind="guessed_date", doc_id=claim_doc.id,
            text=f"We guessed the date of {claim_doc.title} from its text. Is {claim_doc.date} right?",
            choices=["yes", "no"],
        ))
    return result, questions


def subject_of(claim: Claim) -> str:
    """The claim's subject group key. A claim without a subject is its own group (keyed by its text)."""
    return " ".join((claim.subject or claim.text).split()).lower()


def _answer_groups(claims: list[Claim], verdicts: dict[str, Verdict]) -> dict[str, list[Claim]]:
    """Subject groups of the important claims, in claim order: subject key -> all its checked claims."""
    keys: list[str] = []
    for c in claims:
        if c.important and c.id in verdicts and subject_of(c) not in keys:
            keys.append(subject_of(c))
    return {k: [c for c in claims if c.id in verdicts and subject_of(c) == k] for k in keys}


def _is_blocker(v: Verdict) -> bool:
    return v.label == "contradicted" and not v.settled_by_person


def answer_blockers(claims: list[Claim], verdicts: dict[str, Verdict]) -> list[str]:
    """Ids of unsettled contradicted claims in the answer's subject group(s): these stop that part of the answer."""
    return [c.id for members in _answer_groups(claims, verdicts).values()
            for c in members if _is_blocker(verdicts[c.id])]


def choose_answer(
    claims: list[Claim], verdicts: dict[str, Verdict]
) -> tuple[Literal["answered", "not_enough_evidence"], list[str], str | None]:
    """Pick the claims that answer the question, one subject group at a time.

    Only claims in the subject group(s) of the important claims count. Within a group: an unsettled
    contradiction blocks the group; else verified important claims answer; else single-source ones do, with a
    caveat. Groups never affect each other, except that a blocked group is named in the caveat.
    """
    groups = _answer_groups(claims, verdicts)
    multi = len(groups) > 1
    chosen: set[str] = set()
    caveats: list[str] = []
    blocked: list[str] = []
    for members in groups.values():
        name = members[0].subject or members[0].text
        if any(_is_blocker(verdicts[c.id]) for c in members):
            blocked.append(name)
            continue
        important = [c for c in members if c.important]
        verified = [c for c in important if verdicts[c.id].label == "verified"]
        if verified:
            chosen.update(c.id for c in verified)
            continue
        single = [c for c in important if verdicts[c.id].label == "single_source"]
        if not single:
            continue
        chosen.update(c.id for c in single)
        caveat = f"Only one document states this about {name}." if multi else "Only one document states this."
        beat = next((verdicts[c.id].beat_doc_title for c in single if verdicts[c.id].beat_doc_title), None)
        if beat:
            caveat += f" It replaced {beat}'s figure."
        caveats.append(caveat)
    if not chosen:
        return "not_enough_evidence", [], None
    if blocked:
        caveats.append(f"The documents disagree about {', '.join(blocked)}, so that part is not answered.")
    return "answered", [c.id for c in claims if c.id in chosen], " ".join(caveats) or None

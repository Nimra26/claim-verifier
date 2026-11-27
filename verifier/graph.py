"""The claim-verifier LangGraph graph.

doc_facts -> extract (one per document, Send) -> group -> verify (one per claim, Send)
-> label <-> ask_person (interrupt) -> report.

Person answers take effect by re-labelling (Ruling 7): `label` recomputes every verdict from
`checks` + `person_answers` on each pass; `ask_person` only records an answer.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from verifier.models import (
    Answer,
    Claim,
    ClaimCheck,
    DocumentInfo,
    Judgment,
    Passage,
    PersonQuestion,
    Report,
    Verdict,
)
from verifier.reasoner import Reasoner
from verifier.rules import choose_answer, label_claim, needs_recheck
from verifier.search import EvidenceSearch

MAX_TRIES = 3  # one judge call plus at most two re-checks
RECURSION_LIMIT = 500
FAILED_PREFIX = "check failed: "
NOT_ENOUGH = "Not enough evidence to answer."
COULD_NOT_COMPLETE = "The check could not be completed: "
LOW_CONFIDENCE_CAVEAT = "The check on this was low confidence and no person confirmed it."


class State(TypedDict, total=False):
    question: str
    documents: dict[str, DocumentInfo]
    texts: dict[str, str]
    claims: Annotated[list[Claim], operator.add]
    dropped_non_factual: Annotated[int, operator.add]
    subjects: dict[str, str]  # claim id -> subject, set by `group` (claims is append-only)
    checks: Annotated[list[ClaimCheck], operator.add]
    verdicts: dict[str, Verdict]
    pending: list[PersonQuestion]
    person_answers: dict[str, str]
    asked: Annotated[list[PersonQuestion], operator.add]  # questions a person answered
    report: Report | None
    failures: Annotated[list[str], operator.add]  # one plain line per document step that failed


def _with_subjects(state: State) -> list[Claim]:
    """Claims in document order then claim number, with the subject from `group` applied."""
    order = {doc_id: i for i, doc_id in enumerate(state["documents"])}
    subjects = state.get("subjects") or {}

    def key(c: Claim):
        n = c.id.rsplit("#", 1)[-1]
        return (order.get(c.doc_id, len(order)), int(n) if n.isdigit() else 0, c.id)

    return [
        c.model_copy(update={"subject": subjects[c.id]}) if c.id in subjects else c
        for c in sorted(state.get("claims") or [], key=key)
    ]


def build_graph(reasoner: Reasoner, search: EvidenceSearch, *, checkpointer=None):
    async def doc_facts(state: State) -> dict:
        docs = state["documents"]
        texts = state["texts"]
        updated: dict[str, DocumentInfo] = {}
        failures: list[str] = []
        for doc_id, doc in docs.items():
            others = {i: d.title for i, d in docs.items() if i != doc_id}
            try:
                facts = await reasoner.doc_facts(doc.title, texts.get(doc_id, ""), others)
            except Exception as exc:  # noqa: BLE001 - keep the manifest facts and carry on
                failures.append(f"Could not read date or replaces from {doc.title}: {type(exc).__name__}")
                updated[doc_id] = doc
                continue
            change: dict = {}
            if doc.date is None and facts.date is not None:
                change.update(date=facts.date, date_source="text")
            replaces = list(doc.replaces)
            replaces += [r for r in facts.replaces if r not in replaces and r != doc_id]
            change["replaces"] = replaces
            updated[doc_id] = doc.model_copy(update=change)
        return {"documents": updated, "person_answers": state.get("person_answers") or {},
                "failures": failures}

    def to_extract(state: State):
        if not state["documents"]:
            return "group"
        return [
            Send("extract", {"question": state["question"], "doc": doc,
                             "text": state["texts"].get(doc_id, "")})
            for doc_id, doc in state["documents"].items()
        ]

    async def extract(payload: dict) -> dict:
        doc: DocumentInfo = payload["doc"]
        try:
            res = await reasoner.extract_claims(payload["question"], doc, payload["text"])
        except Exception as exc:  # noqa: BLE001 - this document yields no claims; the run continues
            return {"failures": [f"Could not read claims from {doc.title}: {type(exc).__name__}"]}
        claims = [
            Claim(id=f"{doc.id}#{n}", doc_id=doc.id, text=c.text, quote=c.quote, important=c.important)
            for n, c in enumerate(res.claims, start=1)
        ]
        return {"claims": claims, "dropped_non_factual": res.dropped_non_factual}

    async def group(state: State) -> dict:
        claims = _with_subjects(state)
        if not claims:
            return {"subjects": {}}
        try:
            subjects = await reasoner.group_claims(state["question"], claims)
        except Exception as exc:  # noqa: BLE001 - each claim becomes its own subject; the run continues
            return {"subjects": {c.id: c.text for c in claims},
                    "failures": [f"Could not group claims by subject: {type(exc).__name__}"]}
        # A claim the model left out is its own subject group.
        return {"subjects": {c.id: subjects.get(c.id) or c.text for c in claims}}

    def to_verify(state: State):
        claims = _with_subjects(state)
        if not claims:
            return "label"
        return [Send("verify", {"claim": c}) for c in claims]

    async def verify(payload: dict) -> dict:
        claim: Claim = payload["claim"]
        queries: list[str] = []
        passages: dict[str, Passage] = {}
        tries = 0

        async def run_search(query: str) -> None:
            queries.append(query)
            for p in await search.search(query):
                passages.setdefault(p.id, p)

        try:
            await run_search(claim.text)
            if claim.subject and claim.subject != claim.text:
                await run_search(claim.subject)
            tries += 1
            judgment = await reasoner.judge(claim, list(passages.values()))
            while needs_recheck(judgment) and tries < MAX_TRIES:
                await run_search(await reasoner.reword(claim, list(queries)))
                tries += 1
                judgment = await reasoner.judge(claim, list(passages.values()))
        except Exception as exc:  # noqa: BLE001 - a failed check must still give the claim a verdict
            tries = max(tries, 1)
            judgment = Judgment(stances=[], states_claim=True, confidence="low",
                                reason=f"{FAILED_PREFIX}{type(exc).__name__}")
        check = ClaimCheck(claim_id=claim.id, queries=queries, passages=list(passages.values()),
                           judgment=judgment, tries=tries)
        return {"checks": [check]}

    def label(state: State) -> dict:
        docs = state["documents"]
        texts = state["texts"]
        answers = state.get("person_answers") or {}
        by_title = {d.title: d.id for d in docs.values()}
        pair_winner: dict[str, str] = {}
        for qid, ans in answers.items():
            if qid.startswith("replaces:"):
                ans = ans.strip()
                pair_winner[qid] = "neither" if ans.lower() == "neither" else by_title.get(ans, ans)
        checks = {c.claim_id: c for c in state.get("checks") or []}
        verdicts: dict[str, Verdict] = {}
        pending: list[PersonQuestion] = []
        seen: set[str] = set()
        for claim in _with_subjects(state):
            check = checks.get(claim.id)
            if check is None:
                verdicts[claim.id] = Verdict(claim_id=claim.id, label="no_evidence",
                                             reason="could not be checked", evidence=[])
                continue
            v, questions = label_claim(claim, check, docs, texts.get(claim.doc_id, ""),
                                       pair_winner=pair_winner)
            said = answers.get(f"confidence:{claim.id}", "").strip().lower()
            if check.judgment.reason.startswith(FAILED_PREFIX):
                # Ruling 9: a failed check never counts as evidence, unless a person vouches for the claim.
                exc = check.judgment.reason[len(FAILED_PREFIX):]
                if said == "true":
                    v = Verdict(claim_id=claim.id, label="single_source", evidence=[], settled_by_person=True,
                                reason=f"a person said this is true (could not be checked ({exc}))")
                else:
                    v = Verdict(claim_id=claim.id, label="no_evidence", evidence=[],
                                reason=f"could not be checked ({exc})")
            elif said == "true":
                v = v.model_copy(update={"settled_by_person": True})
            if said == "false":
                v = v.model_copy(update={"label": "contradicted", "settled_by_person": True,
                                         "reason": "a person said this is false"})
            verdicts[claim.id] = v
            for q in questions:
                if q.id not in answers and q.id not in seen:
                    seen.add(q.id)
                    pending.append(q)
        return {"verdicts": verdicts, "pending": pending}

    def after_label(state: State) -> str:
        return "ask_person" if state.get("pending") else "report"

    def ask_person(state: State) -> dict:
        question = state["pending"][0]
        answer = interrupt(question.model_dump())
        answer = str(answer).strip()
        update: dict = {"person_answers": {**(state.get("person_answers") or {}), question.id: answer},
                        "asked": [question]}
        if question.kind == "guessed_date" and answer.lower() == "no" and question.doc_id in state["documents"]:
            docs = dict(state["documents"])
            docs[question.doc_id] = docs[question.doc_id].model_copy(update={"date": None, "date_source": None})
            update["documents"] = docs
        return update

    async def report(state: State) -> dict:
        claims = _with_subjects(state)
        verdicts = state.get("verdicts") or {}
        failures = list(state.get("failures") or [])
        # I1: no claims at all while model steps failed means the check never really ran.
        completed = not (failures and not claims)
        status, claim_ids, caveat = choose_answer(claims, verdicts)
        if not completed:
            status, claim_ids, caveat = "not_enough_evidence", [], None
            text = COULD_NOT_COMPLETE + "; ".join(failures)
        elif status == "answered":
            chosen = [c for c in claims if c.id in claim_ids]
            checks_by_id = {c.claim_id: c for c in state.get("checks") or []}
            if any(c.id in checks_by_id and checks_by_id[c.id].judgment.confidence == "low"
                   and not verdicts[c.id].settled_by_person for c in chosen):
                caveat = f"{caveat} {LOW_CONFIDENCE_CAVEAT}" if caveat else LOW_CONFIDENCE_CAVEAT
            try:
                text = await reasoner.write_answer(state["question"], chosen,
                                                   [verdicts[c.id] for c in chosen], caveat)
            except Exception as exc:  # noqa: BLE001 - fall back to the chosen claims' own words
                failures.append(f"Could not write the answer: {type(exc).__name__}")
                text = " ".join(c.text for c in chosen)
        else:
            text = NOT_ENOUGH
        order = {c.id: i for i, c in enumerate(claims)}
        checks = sorted(state.get("checks") or [], key=lambda c: order.get(c.claim_id, len(order)))
        return {"report": Report(
            question=state["question"],
            answer=Answer(status=status, text=text, caveat=caveat, claim_ids=claim_ids),
            documents=list(state["documents"].values()),
            claims=claims,
            checks=checks,
            verdicts=[verdicts[c.id] for c in claims if c.id in verdicts],
            person_answers=dict(state.get("person_answers") or {}),
            asked=list(state.get("asked") or []),
            dropped_non_factual=state.get("dropped_non_factual") or 0,
            failures=failures,
            completed=completed,
        )}

    g = StateGraph(State)
    g.add_node("doc_facts", doc_facts)
    g.add_node("extract", extract)
    g.add_node("group", group)
    g.add_node("verify", verify)
    g.add_node("label", label)
    g.add_node("ask_person", ask_person)
    g.add_node("report", report)
    g.add_edge(START, "doc_facts")
    g.add_conditional_edges("doc_facts", to_extract, ["extract", "group"])
    g.add_edge("extract", "group")
    g.add_conditional_edges("group", to_verify, ["verify", "label"])
    g.add_edge("verify", "label")
    g.add_conditional_edges("label", after_label, ["ask_person", "report"])
    g.add_edge("ask_person", "label")
    g.add_edge("report", END)
    # Each person question costs two steps (label + ask_person); lift the default limit of 25.
    return g.compile(checkpointer=checkpointer).with_config(recursion_limit=RECURSION_LIMIT)

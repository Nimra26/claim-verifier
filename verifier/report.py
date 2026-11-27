"""Turn a Report into plain-words Markdown plus the raw JSON."""

from collections import Counter
from pathlib import Path

from verifier.models import Claim, Passage, Report, Verdict
from verifier.rules import answer_blockers

LABEL_WORDS = {
    "verified": "Verified",
    "single_source": "Only one document says this",
    "contradicted": "Documents disagree",
    "outdated": "Outdated",
    "no_evidence": "No evidence",
}
QUOTE_LIMIT = 200


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def _quote(passage: Passage, titles: dict[str, str]) -> str:
    text = " ".join(passage.text.split())
    if len(text) > QUOTE_LIMIT:
        text = text[:QUOTE_LIMIT] + "…"
    title = titles.get(passage.doc_id, passage.source)
    where = f"{title}, p. {passage.page}" if passage.page is not None else title
    return f"“{text}” ({where})"


def _evidence(report: Report, verdict: Verdict, titles: dict[str, str]) -> list[str]:
    passages = {}
    for check in report.checks:
        if check.claim_id == verdict.claim_id:
            passages = {p.id: p for p in check.passages}
    return [_quote(passages[i], titles) for i in verdict.evidence if i in passages]


def _evidence_cell(report: Report, verdict: Verdict, claim: Claim, titles: dict[str, str]) -> list[str]:
    """What the Evidence column shows. A single-source claim's evidence is its own quote (I3); any passages
    it beat are shown as what the older document said."""
    quotes = _evidence(report, verdict, titles)
    if verdict.label != "single_source":
        return quotes
    own = f"Stated in {titles.get(claim.doc_id, claim.doc_id)}: “{' '.join(claim.quote.split())}”"
    return [own] + [f"Older document said: {q}" for q in quotes]


def _low_confidence(report: Report, verdict: Verdict) -> bool:
    """Judge confidence stayed low and no person settled the claim (I4)."""
    if verdict.settled_by_person:
        return False
    return any(c.claim_id == verdict.claim_id and c.judgment.confidence == "low" for c in report.checks)


def _source(doc) -> str:
    if doc is None:
        return ""
    return f"{doc.title}, {doc.date}" if doc.date else doc.title


def render_markdown(report: Report) -> str:
    titles = {d.id: d.title for d in report.documents}
    docs = {d.id: d for d in report.documents}
    claims: dict[str, Claim] = {c.id: c for c in report.claims}
    out = [f"# {report.question}", "", "## Answer", ""]

    ans = report.answer
    if ans.status == "answered":
        out.append(ans.text)
    elif not report.completed:
        out.append(ans.text)
    else:
        out.append("Not enough evidence to answer.")
        out.append("")
        verdicts = {v.claim_id: v for v in report.verdicts}
        if answer_blockers(report.claims, verdicts):
            out.append("The documents disagree and nothing settled which one is right.")
        else:
            out.append("No checked statement in the documents answers this question.")
    if ans.caveat:
        out += ["", f"*{ans.caveat}*"]

    out += ["", "## Claims", "", "| Claim | Label | Why | Source | Evidence |", "| --- | --- | --- | --- | --- |"]
    for v in report.verdicts:
        c = claims.get(v.claim_id)
        if c is None:
            continue
        ev = "<br>".join(_cell(q) for q in _evidence_cell(report, v, c, titles))
        label = LABEL_WORDS[v.label] + (" (low confidence)" if _low_confidence(report, v) else "")
        out.append(
            f"| {_cell(c.text)} | {label} | {_cell(v.reason)} | "
            f"{_cell(_source(docs.get(c.doc_id)))} | {ev} |"
        )

    out += ["", "## Contradictions", ""]
    found = False
    for v in report.verdicts:
        if v.label not in ("contradicted", "outdated"):
            continue
        c = claims.get(v.claim_id)
        if c is None:
            continue
        found = True
        out.append(f"- **{LABEL_WORDS[v.label]}:** {c.text}")
        out.append(f"  - This document says: “{' '.join(c.quote.split())}” ({_source(docs.get(c.doc_id))})")
        for q in _evidence(report, v, titles):
            out.append(f"  - The other side says: {q}")
        if v.beat_doc_title:
            out.append(f"  - It replaced {v.beat_doc_title}'s figure.")
    if not found:
        out.append("None.")

    if report.asked:
        out += ["", "## Settled by a person", ""]
        for q in report.asked:
            out.append(f"- {q.text} → {report.person_answers.get(q.id, '')}")

    if report.failures:
        out += ["", "## Could not check", ""]
        out += [f"- {f}" for f in report.failures]

    counts = Counter(v.label for v in report.verdicts)
    parts = [f"{LABEL_WORDS[k]}: {counts.get(k, 0)}" for k in LABEL_WORDS]
    out += ["", "---", "", " · ".join(parts) + f" · Dropped as not factual: {report.dropped_non_factual}", ""]
    return "\n".join(out)


def write_report(report: Report, folder: Path) -> tuple[Path, Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    md_path, json_path = folder / "report.md", folder / "report.json"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return md_path, json_path

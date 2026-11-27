import datetime as dt
from typing import Literal

from pydantic import BaseModel

Label = Literal["verified", "single_source", "contradicted", "outdated", "no_evidence"]


class DocumentInfo(BaseModel):
    id: str
    file: str
    title: str
    date: dt.date | None = None
    date_source: Literal["manifest", "text"] | None = None
    replaces: list[str] = []


class Claim(BaseModel):
    id: str
    doc_id: str
    text: str
    quote: str
    important: bool
    subject: str = ""


class Passage(BaseModel):
    id: str
    doc_id: str
    source: str
    text: str
    page: int | None = None
    score: float


class Stance(BaseModel):
    passage_id: str
    stance: Literal["supports", "contradicts", "unrelated"]


class Judgment(BaseModel):
    stances: list[Stance]
    states_claim: bool
    confidence: Literal["high", "medium", "low"]
    reason: str


class ClaimCheck(BaseModel):
    claim_id: str
    queries: list[str]
    passages: list[Passage]
    judgment: Judgment
    tries: int


class Verdict(BaseModel):
    claim_id: str
    label: Label
    reason: str
    evidence: list[str]
    settled_by_person: bool = False
    beat_doc_title: str | None = None


class PersonQuestion(BaseModel):
    id: str
    kind: Literal["low_confidence", "replaces_vs_date", "guessed_date"]
    claim_id: str | None = None
    doc_id: str | None = None
    text: str
    choices: list[str]


class Answer(BaseModel):
    status: Literal["answered", "not_enough_evidence"]
    text: str
    caveat: str | None = None
    claim_ids: list[str]


class Report(BaseModel):
    question: str
    answer: Answer
    documents: list[DocumentInfo]
    claims: list[Claim]
    checks: list[ClaimCheck]
    verdicts: list[Verdict]
    person_answers: dict[str, str]
    dropped_non_factual: int
    failures: list[str] = []
    asked: list[PersonQuestion] = []
    completed: bool = True  # False when model failures meant the check could not run (I1)

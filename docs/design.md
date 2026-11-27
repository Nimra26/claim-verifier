# Claim Verifier — design (R&D prototype)

Date: 26 Nov 2026. Branch: `claim-verifier`. Status: draft for approval.

## Intent and the why

AI answers sound convincing even when the information behind them is incomplete, outdated, contradictory across
documents, or has no evidence at all. RAG retrieves passages, but retrieval does not make the answer correct.

The Claim Verifier is a LangGraph agent that, given a question and several documents on the same topic (some
correct, some outdated, some contradicting each other):

1. reads every document,
2. pulls out each factual claim that bears on the question,
3. searches for evidence for every claim using Laraib's existing retrieval (the Ask the Workspace RAG pipeline),
4. labels each claim: **verified**, **single source**, **contradicted**, **outdated**, or **no evidence**,
5. re-checks the claims it is unsure about,
6. asks a person to settle ambiguities that matter,
7. returns a verified report with evidence references, and says plainly when there is not enough evidence to
   answer.

Target users: anyone who needs an answer from a set of documents they can trust.

## Goals

- Every extracted claim gets a verdict. No claim is silently dropped.
- Every verdict carries the evidence behind it (document, page when known, the passage text).
- Contradictions name both sides and both sources.
- "Not enough evidence" is a first-class answer, not a failure.
- Proves the idea end to end on a made-up test set that contains correct, outdated and contradictory documents.

## Non-goals (this prototype)

- Not part of the Ask the Workspace app: no UI page, no route, no admin setting, no client install.
- No real (client or Taleemabad) documents. Made-up documents only, because the model is a cloud model.
- No OCR, no Urdu/Roman Urdu handling, no access groups (one user, all test documents visible).
- No new database tables or Alembic migrations.

## Anti-goals (a failure even if it runs)

- A report that reads confident while the evidence is thin. Uncertainty must show.
- A verdict without the passage it rests on.
- Skipping claims because the run got long.
- Changing `src/modules/workspace/service.py` or any other app behaviour to make the prototype work.
- Touching the live database `agentic_rag_db`.

## Constraints

- Lives in `ask-the-workspace/verifier/`, imports the app's retrieval code; does not copy it.
- Retrieval reuse point: `RAGRetrievalService.retrieve(query, top_k, doc_ids)` (`src/rag/pipeline/services.py`) —
  hybrid pgvector + keyword search, optional rerank — and `RAGIngestionService` to index the test documents.
  Not `WorkspaceAskService._search`: it is private and tied to translation, chat history and the access filter.
- Storage: a throwaway pgvector Postgres container (`--restart no`, named `claim-verifier-db`, removed after use).
- Embeddings: Ollama `bge-m3`, as the app uses, called through the app's `OpenAIEmbeddingProvider` at
  `{OLLAMA}/v1`. Not on this laptop yet: the owner downloads it after 8 pm, 26 Nov. Until then everything is built
  and tested with fake embeddings; only the real run waits.
- Database: the Supabase Postgres image already on this laptop (`public.ecr.aws/supabase/postgres:17.6.1.165`,
  ships pgvector), container `claim-verifier-db`; no image download. Only Ollama runs locally; the agent's reasoning model is
  remote.
- Reasoning model: Claude through the API, behind one small interface so a local Ollama model can be swapped in.
  *(27 Nov: the owner has no API key, so the default route is the local Claude Code command with its login — `claude -p`
  with structured output, no tools — needing no new package; the API route stays behind `VERIFIER_BACKEND=api`.)*
  **New package: `anthropic` (the SDK, used directly; replaced the first plan of `langchain-anthropic`, Ruling 13, 27 Nov)** — to be confirmed with the owner before installing
  (repo rule: no new Python packages without saying so). `langgraph` is already in `uv.lock` (1.1.3) via `langchain`;
  it becomes a direct dependency.
- Human-in-the-loop uses LangGraph `interrupt()` with the in-memory checkpointer (`MemorySaver`, already in
  `langgraph-checkpoint`). The run is one terminal session; no resume across restarts.
- Slow internet (~105 KB/s): only the two packages above.

## Approaches considered

- **A. Fixed step-by-step graph (picked).** Every claim follows the same path (search → judge → re-check →
  ask a person), fanned out in parallel with LangGraph `Send`. Predictable, every step testable, no claim skipped.
  Cost: less flexible; a new kind of case means a new node.
- **B. One free-roaming tool-using agent** (search, flag, ask person). Quicker to write, but on many claims it skips
  some or stops early: the exact failure we are fixing. Rejected.
- **C. Pairwise document comparison**, no per-claim search. Ignores the retrieval pipeline, grows quadratically
  with documents, cannot tell "supported elsewhere" from "unsupported". Rejected.

Owner picked A (26 Nov).

## Decisions made with the owner

- Separate prototype, not an app feature (26 Nov).
- Claude via API on made-up documents; swappable for a local model later (26 Nov, proposed and accepted).
- Asking a person happens in the terminal for now (26 Nov, proposed and accepted).
- **Outdated rule (26 Nov):** each document has a date. When two claims conflict, the newer document wins and the
  older claim is labelled *outdated*. A document that says it replaces another ("this replaces the 2023 policy")
  wins. When the two point different ways — a document says it replaces one with a *newer* date — the agent asks
  a person instead of applying either rule. *(Builder check, 26 Nov: the first wording said both "always wins" and
  "ask a person"; settled as written here, my call, because asking is the safe side.)*
- Input is a question plus a set of documents; the report answers the question.
- **How cautious the answer is (26 Nov, owner: option a).** The report answers from the best claim it has and puts
  the caveat beside it: a *single source* or *outdated-replaced* answer is given, with a visible note ("only one
  document states this; it replaced the 2023 figure of 20"). It says "not enough evidence to answer" only when the
  claim that would answer the question has *no evidence*, or its *contradiction* was not settled (by the rule or by
  a person).
- `langchain-anthropic` may be installed (26 Nov, owner), in the `verifier` dependency group only. Built with the `anthropic` SDK directly instead (27 Nov): the model rejects the forced-tool requests langchain-anthropic 1.2.0 sends; same vendor, smaller footprint.

## The design

### Input

A folder of documents plus a `manifest.yaml`:

```yaml
question: "How many days of annual leave does a full-time employee get?"
documents:
  - file: leave-policy-2023.md
    date: 2023-01-15
  - file: leave-policy-2025.md
    date: 2025-03-01
  - file: hr-faq.md
    date: 2024-06-10
```

`date` is the date the document was written or issued. If missing, the agent tries to read it from the text and
asks a person to confirm it; if none is found the document has no date and can never make another *outdated*.

### Graph

```
load_and_index ─▶ extract_claims (one per document, parallel)
               ─▶ group_claims  (same-subject claims across documents)
               ─▶ verify_claim  (one per claim group, parallel via Send):
                     search_evidence ─▶ judge
                        judge confidence low and tries < 2 ─▶ reword query ─▶ search_evidence
                        still low and important ─▶ ask_person (interrupt)
               ─▶ resolve_conflicts (apply the outdated rule; ask a person on a dates-vs-"replaces" clash)
               ─▶ write_report
```

State (one `TypedDict`): question, documents (id, file, date, replaces[]), claims (id, text, doc_id, location,
subject), evidence per claim (passages with doc_id, page, text, score), verdict per claim (label, confidence,
reason, evidence ids, tries), questions asked and the person's answers, report.

### Nodes

- **load_and_index** — read the manifest, index each document with `RAGIngestionService` into the throwaway
  pgvector store, record its date. Also extract any "replaces / supersedes" statements per document.
- **extract_claims** — the model reads one whole document and returns atomic factual claims relevant to the
  question, each with the quote it came from. Structured output (pydantic). Claims that are opinion or
  instructions are dropped and counted.
- **group_claims** — claims about the same subject across documents are grouped (e.g. "annual leave days for
  full-time staff"), so contradictions surface as one group with differing values.
- **search_evidence** — `retrieve()` over all test documents with the claim as query (top_k ~8), plus a second
  query built from the subject. Evidence from the claim's own source passage does not count as independent support.
- **judge** — the model reads the claim and the passages and returns: supports / contradicts / unrelated per
  passage, an overall label, a confidence (high/medium/low) and a one-line reason citing passage ids.
- **re-check** — on low confidence, the model rewrites the search query and the claim is searched and judged
  again, at most 2 extra tries.
- **ask_person** — `interrupt()` with the claim, the competing passages and the choices; the person's answer is
  stored as the verdict with reason "settled by a person". Triggered when: confidence is still low after re-checks
  **and** the claim is important; or a dates-vs-"replaces" clash; or a document's date was guessed.
  *Important* = the claim directly answers the question (the model marks this at extraction).
- **resolve_conflicts** — within a group whose claims disagree: apply the outdated rule; anything the rule cannot
  decide (same date, no date, no "replaces") is **contradicted** and, if important, goes to a person.
- **write_report** — Markdown report (and the same data as JSON): the answer to the question first (or "not enough
  evidence to answer", with why); then a table of claims with label, sources, evidence quotes and page; then a
  contradictions section showing both sides; then what a person settled.

### Labels

| Label | Means |
|---|---|
| verified | supported by at least one passage from a *different* document, none contradicting it from a newer or equal-date document |
| contradicted | a passage of equal or unknown recency says otherwise and no rule decides |
| outdated | a newer document (or one that replaces it) says otherwise |
| single source | its own document states it (its quote is found in that document), nothing from another document supports it, and nothing of equal or newer date contradicts it (an older document may disagree: that one is *outdated*) *(added after the builder check, my call)* |
| no evidence | the claim cannot be traced to a passage at all: its quote is not in its document, or the judge finds its own passage does not state it. Also the answer's state when **no** claim bears on the question *(my call, 26 Nov: the first wording overlapped with single source)* |

### Success test

A made-up set of 4–6 documents on one topic, built for the test, with a known answer key: at least one outdated
claim, one plain contradiction, one claim with no support, one claim verified by two documents, and one
dates-vs-"replaces" clash. Plus one question whose answer is not in the documents. The run passes when every key
claim gets the expected label and the no-answer question returns "not enough evidence".

Two levels *(my call after the builder check)*: node logic (rules, labelling, report) gets unit tests with a fake
model — the gate for each code change. The **acceptance check** is the real Claude run on the answer key, with the
person's answers scripted from a file (`--answers answers.yaml`) so it can be repeated; prompts are tuned until it
passes, up to a small budget of runs, and any label that still misses is reported, not hidden.

Note: `retrieve()` needs explicit `doc_ids` (pgvector refuses an unfiltered search); pass all test document ids.
Markdown inputs have no page numbers: evidence cites the passage and its heading instead.

### Running it

`uv run python -m verifier run <folder>` — indexes, runs the graph, pauses in the terminal for any question to a
person, writes `report.md` and `report.json` into the folder.

## Left to the builder

- Prompt wording, exact pydantic schemas, top_k, and the claim-grouping method (model call vs embedding similarity).
- Which Claude model (default to the current Sonnet for cost; Opus if quality falls short).
- File layout inside `verifier/`.
- How the throwaway database and Ollama embedding are wired (settings via env, never `.env` of the app).

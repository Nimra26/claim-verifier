# Claim Verifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A standalone LangGraph agent that checks every factual claim in a set of documents against evidence found with the app's own retrieval, and writes a verified report.

**Architecture:** A fixed LangGraph graph (`verifier/graph.py`) fans claims out with `Send`, checks each one (search → judge → re-check up to 2 times), applies pure-Python labelling rules (`verifier/rules.py`), asks a person through `interrupt()` one question at a time, and writes Markdown + JSON. The model sits behind a `Reasoner` protocol (Claude now, swappable); search goes through `EvidenceSearch`, a thin wrapper around the app's `RAGRetrievalService`. Nothing under `src/` changes.

**Tech Stack:** Python 3.12+ (use the installed 3.13: `UV_PYTHON=3.13`), langgraph 1.1.3 (already locked), langchain-anthropic (new, owner approved 26 Nov), pydantic v2, the app's `RAGIngestionService` / `RAGRetrievalService` / `PgVectorStore` / `OpenAIEmbeddingProvider`, Ollama `bge-m3`, Postgres with pgvector (Supabase image already local).

**Spec:** `docs/superpowers/specs/2026-11-26-claim-verifier-design.md` — read it before Task 1.

## Global Constraints

- All new code lives in `verifier/` at the repo root. **No file under `src/` changes.** No verifier import of `anthropic`/`langchain_anthropic` from `src/`.
- `langgraph` and `langchain-anthropic` go in a new `[dependency-groups] verifier` list in `pyproject.toml`, never `[project].dependencies`. `pyyaml` is already in the lock (via other packages) — list it in the `verifier` group too so the import is declared.
- No other new packages. Slow internet (~105 KB/s).
- Labels, exactly: `verified`, `single_source`, `contradicted`, `outdated`, `no_evidence`.
- Re-check: at most **2** extra tries per claim (3 judge calls total).
- Evidence search: `top_k = 8` per query; a passage from the claim's own document never counts as support or contradiction.
- Never touch `agentic_rag_db`; never open the app's `.env`. Verifier settings come from env vars prefixed `VERIFIER_` plus `ANTHROPIC_API_KEY`.
- Database container: `claim-verifier-db`, image `public.ecr.aws/supabase/postgres:17.6.1.165`, `--restart no`, removed at the end.
- Unit tests run offline with fakes: `UV_PYTHON=3.13 uv run --group verifier --group dev pytest -q -p no:cacheprovider verifier/tests`. One test process at a time (CLAUDE.md "Machine load").
- Ollama + `bge-m3` arrive after 8 pm 26 Nov. Tasks 1–8 need neither; Task 9's real run does.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Branch `claim-verifier`; never push.

## Review Focus

1. **A document's claim quote that is not literally in its text** (model paraphrased it) — must become `no_evidence`, not `single_source`. Test in Task 3 (`test_quote_not_in_document_is_no_evidence`) and Task 2 (`quote_in_document` tolerates whitespace/case/curly quotes).
2. **A question nothing in the documents answers** — report says "not enough evidence", never a guessed answer. Test in Task 3 (`test_answer_none_when_no_important_claim`) and Task 9 testset.
3. **A model call that fails or returns invalid structure** for one claim — that claim gets `confidence="low"` with reason "check failed", the run continues; no claim vanishes. Test in Task 6 (`test_failed_judge_does_not_drop_claim`).
4. **Re-running on the same folder** — old chunks for each doc id are deleted before re-indexing, so evidence is not doubled. Test in Task 4 (`test_index_replaces_previous_chunks`).
5. **Document with no date in manifest and none in text** — never makes another outdated; conflicts with it are `contradicted`. Test in Task 3 (`test_undated_never_outdates`).

---

### Task 1: Package skeleton, data models, manifest

**Files:**
- Modify: `pyproject.toml` (add `verifier` dependency group), `uv.lock` (via `uv lock`)
- Create: `verifier/__init__.py`, `verifier/models.py`, `verifier/manifest.py`, `verifier/tests/__init__.py`, `verifier/tests/conftest.py`
- Test: `verifier/tests/test_manifest.py`

**Interfaces:**
- Produces (`verifier/models.py`, all pydantic `BaseModel`):
  - `Label = Literal["verified","single_source","contradicted","outdated","no_evidence"]`
  - `DocumentInfo(id: str, file: str, title: str, date: date | None, date_source: Literal["manifest","text"] | None, replaces: list[str] = [])` — `id` = file stem; `replaces` holds doc ids.
  - `Claim(id: str, doc_id: str, text: str, quote: str, important: bool, subject: str = "")` — `id` = `f"{doc_id}#{n}"`, n from 1.
  - `Passage(id: str, doc_id: str, source: str, text: str, page: int | None, score: float)` — `id` = `f"{doc_id}:{chunk_id}"`.
  - `Stance(passage_id: str, stance: Literal["supports","contradicts","unrelated"])`
  - `Judgment(stances: list[Stance], states_claim: bool, confidence: Literal["high","medium","low"], reason: str)` — `states_claim`: the claim's own passage really says it.
  - `ClaimCheck(claim_id: str, queries: list[str], passages: list[Passage], judgment: Judgment, tries: int)`
  - `Verdict(claim_id: str, label: Label, reason: str, evidence: list[str], settled_by_person: bool = False)`
  - `PersonQuestion(id: str, kind: Literal["low_confidence","replaces_vs_date","guessed_date"], claim_id: str | None, doc_id: str | None, text: str, choices: list[str])`
  - `Answer(status: Literal["answered","not_enough_evidence"], text: str, caveat: str | None, claim_ids: list[str])`
  - `Report(question: str, answer: Answer, documents: list[DocumentInfo], claims: list[Claim], checks: list[ClaimCheck], verdicts: list[Verdict], person_answers: dict[str, str], dropped_non_factual: int)`
- Produces (`verifier/manifest.py`): `Manifest(question: str, documents: list[DocumentInfo])`; `load_manifest(folder: Path) -> tuple[Manifest, dict[str, str]]` returning the manifest and `{doc_id: text}`. Reads `.md` and `.txt` only; title = first `# ` heading else file stem.

- [ ] **Step 1:** Add to `pyproject.toml`: `verifier = ["langgraph>=1.1.3", "langchain-anthropic>=1.0", "pyyaml"]` under `[dependency-groups]`; run `UV_PYTHON=3.13 uv lock` then `UV_PYTHON=3.13 uv sync --group verifier --group dev`. Expected: lock adds `langchain-anthropic`, `anthropic` (and their small deps) only; `git diff --stat uv.lock` shows no change to existing package versions. If it wants to upgrade existing packages, stop and report.
- [ ] **Step 2:** `verifier/tests/conftest.py` sets, before any `src` import, the dummy env vars from CLAUDE.md's test command (`ACCESS_TOKEN_SECRET_KEY=x`, `REFRESH_TOKEN_SECRET_KEY=x`, `RESET_PASSWORD_TOKEN_SECRET=x`, `VERIFICATION_TOKEN_SECRET=x`, `DATABASE_URL=postgresql+asyncpg://u:p@localhost:1/x`) via `os.environ.setdefault`.
- [ ] **Step 3: Failing tests** in `test_manifest.py`:
  - `test_loads_question_dates_and_texts`: tmp folder with `manifest.yaml` (question + two docs, one with `date: 2023-01-15`, one without) and both files → `manifest.question` matches; doc `a` has `date == date(2023,1,15)` and `date_source == "manifest"`; doc `b` has `date is None`, `date_source is None`; texts dict has both ids.
  - `test_missing_file_raises`: manifest names `nope.md` → `FileNotFoundError` whose message contains `nope.md`.
  - `test_title_from_heading`: file starting `# Leave Policy 2025` → `title == "Leave Policy 2025"`.
- [ ] **Step 4:** Run, expect FAIL (import error). Implement `models.py` and `manifest.py`. Run, expect PASS.
- [ ] **Step 5:** Commit `pyproject.toml uv.lock verifier/` — "verifier: package skeleton, models, manifest".

### Task 2: Text helpers — quote matching and dates

**Files:**
- Create: `verifier/text.py`
- Test: `verifier/tests/test_text.py`

**Interfaces:**
- Produces: `normalise(s: str) -> str` (lowercase, curly→straight quotes, collapse whitespace); `quote_in_document(quote: str, text: str) -> bool` (substring after `normalise`; empty quote → False); `compare_dates(a: date | None, b: date | None) -> Literal["a_newer","b_newer","same","unknown"]`.

- [ ] **Step 1: Failing tests:** `test_quote_match_ignores_case_space_quotes` (`"Staff get  25 “days”"` in `"... staff get 25 \"days\" of leave"` → True); `test_quote_absent` → False; `test_empty_quote_false`; `test_compare_dates` covering all four results (either `None` → `"unknown"`).
- [ ] **Step 2:** Run → FAIL. Implement. Run → PASS. Commit "verifier: quote and date helpers".

### Task 3: Labelling rules (pure Python, the heart of it)

**Files:**
- Create: `verifier/rules.py`
- Test: `verifier/tests/test_rules.py`

**Interfaces:**
- Consumes: models (Task 1), `quote_in_document`, `compare_dates` (Task 2).
- Produces:
  - `supersession(claim_doc: DocumentInfo, other_doc: DocumentInfo) -> Literal["claim_outdated","claim_current","undecided","ask_person"]`
  - `label_claim(claim: Claim, check: ClaimCheck, docs: dict[str, DocumentInfo], doc_text: str) -> tuple[Verdict, list[PersonQuestion]]`
  - `needs_recheck(judgment: Judgment) -> bool` — `confidence == "low"`.
  - `choose_answer(claims: list[Claim], verdicts: dict[str, Verdict]) -> tuple[Literal["answered","not_enough_evidence"], list[str], str | None]` — (status, claim ids the answer rests on, caveat).

`supersession` (exact algorithm, from the spec's outdated rule):
```python
def supersession(claim_doc, other_doc):
    by_date = compare_dates(claim_doc.date, other_doc.date)  # "a_newer" = claim doc newer
    if claim_doc.id in other_doc.replaces:
        return "ask_person" if by_date == "a_newer" else "claim_outdated"
    if other_doc.id in claim_doc.replaces:
        return "ask_person" if by_date == "b_newer" else "claim_current"
    if by_date in ("unknown", "same"):
        return "undecided"
    return "claim_outdated" if by_date == "b_newer" else "claim_current"
```

`label_claim` order (first match wins); `others` = stances on passages whose `doc_id != claim.doc_id`:
1. Quote not in `doc_text`, or `judgment.states_claim is False` → `no_evidence`.
2. For each contradicting other doc: `supersession`. Any `ask_person` → also emit a `PersonQuestion(kind="replaces_vs_date")` (choices: the two doc titles + "neither"), label stays provisional `contradicted`. Any `claim_outdated` → `outdated`. Any `undecided` → `contradicted`.
3. Supporting passage from another doc → `verified`.
4. Else → `single_source`.
- If `judgment.confidence == "low"` and `claim.important` → emit `PersonQuestion(kind="low_confidence")` with choices `["true","false","can't tell"]`.
- If `claim_doc.date_source == "text"` → emit `PersonQuestion(kind="guessed_date", doc_id=...)` once per doc (dedupe by id `f"date:{doc_id}"`).
- `evidence` = ids of the passages that decided the label; `reason` = one plain sentence naming the deciding document(s).

`choose_answer`: among `important` claims, prefer `verified`, then `single_source` (caveat: "Only one document states this." plus, if it beat an older document, "It replaced <older title>'s figure."); if any important claim is `contradicted` and not settled by a person and no `verified` one exists → `not_enough_evidence`; no important claim, or all `no_evidence`/`outdated` → `not_enough_evidence`.

- [ ] **Step 1: Failing tests** (build docs/claims inline; dates 2023-01-15 = "old", 2025-03-01 = "new"):
  - `test_newer_wins`: claim in old, contradicted by new → `outdated`.
  - `test_newer_claim_vs_older_is_single_source`: claim in new, contradicted only by old, no support → `single_source`.
  - `test_replaces_wins_when_dates_agree`: new doc `replaces=["old"]` → old claim `outdated`.
  - `test_replaces_older_dated_than_replaced_asks_person`: old-dated doc says it replaces new-dated → one `replaces_vs_date` question, label `contradicted`.
  - `test_undated_never_outdates`: claim in dated doc contradicted by undated doc → `contradicted`.
  - `test_same_date_conflict_is_contradicted`.
  - `test_support_from_other_doc_is_verified`.
  - `test_own_doc_passage_never_counts`: only supporting passage is from the claim's own doc → `single_source`.
  - `test_quote_not_in_document_is_no_evidence`.
  - `test_low_confidence_important_asks_person` / `test_low_confidence_unimportant_does_not_ask`.
  - `test_guessed_date_asks_once_per_doc`.
  - `test_answer_prefers_verified`, `test_answer_single_source_has_caveat` (caveat contains "Only one document"), `test_answer_unsettled_contradiction_declines`, `test_answer_none_when_no_important_claim`.
- [ ] **Step 2:** Run → FAIL. Implement. Run → PASS. Commit "verifier: labelling rules".

### Task 4: Store and evidence search (reuses the app's retrieval)

**Files:**
- Create: `verifier/config.py`, `verifier/store.py`, `verifier/search.py`
- Test: `verifier/tests/test_search.py`, `verifier/tests/fakes.py`

**Interfaces:**
- Produces:
  - `VerifierSettings` (pydantic-settings, env prefix `VERIFIER_`): `database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:55432/postgres"`, `ollama_url: str = "http://localhost:11434"`, `embed_model: str = "bge-m3"`, `model: str = "claude-sonnet-5-5"`, `top_k: int = 8`, `chunk_size: int = 800`, `chunk_overlap: int = 120`. (`ANTHROPIC_API_KEY` is read by the SDK itself.)
  - `ensure_schema(engine: AsyncEngine) -> None` — `CREATE EXTENSION IF NOT EXISTS vector` + the `rag_chunks` DDL and three indexes copied verbatim from `alembic/versions/b7c1d2e3f4a5_add_rag_chunks_pgvector.py`, each with `IF NOT EXISTS`. Comment the source revision.
  - `Store(ingestion: RAGIngestionService, retrieval: RAGRetrievalService, vector_store: VectorStore)`; `build_store(settings, *, embedding_provider=None, vector_store=None) -> Store` — defaults: `OpenAIEmbeddingProvider(api_key="ollama", model=settings.embed_model, base_url=f"{settings.ollama_url}/v1")`, `PgVectorStore(database_url=settings.database_url)`, `ChunkingStrategyRegistry([FixedWindowChunkingStrategy(), RecursiveSemanticChunkingStrategy()])`, strategy `"recursive_semantic"`, no reranker, `prefetch_k = 2 * top_k`.
  - `async index_documents(store: Store, texts: dict[str, str], titles: dict[str, str]) -> None` — per doc: `vector_store.delete_by_doc_id(doc_id=...)` then `ingestion.ingest_text(text=..., source=title, doc_id=doc_id)`.
  - `EvidenceSearch(retrieval: RAGRetrievalService, doc_ids: list[str], top_k: int)`; `async search(query: str) -> list[Passage]` — calls `retrieve(query=..., top_k=..., doc_ids=...)` and maps `RetrievedChunk` → `Passage` (`id=f"{doc_id}:{chunk_id}"`, `page=page_number`).
- `verifier/tests/fakes.py`: `FakeEmbeddings` (deterministic bag-of-words hash into 1024 floats), `MemoryVectorStore(VectorStore)` implementing `upsert_chunks`, `delete_by_doc_id`, `similarity_search` (cosine, honours `doc_ids`, `supports_hybrid = False`).

- [ ] **Step 1: Failing tests:** `test_search_maps_chunks_to_passages` (index two docs with fakes, search a word only doc `b` has → first passage `doc_id == "b"`, ids look like `"b:..."`); `test_search_limited_to_doc_ids`; `test_index_replaces_previous_chunks` (index same doc twice → chunk count unchanged).
- [ ] **Step 2:** Run → FAIL. Implement. Run → PASS. Commit "verifier: store and evidence search on the app's retrieval".

### Task 5: Reasoner (the model behind one interface)

**Files:**
- Create: `verifier/reasoner.py`, `verifier/prompts.py`
- Modify: `verifier/tests/fakes.py` (add `FakeReasoner`)
- Test: `verifier/tests/test_reasoner.py`

**Interfaces:**
- Produces `Reasoner` (a `typing.Protocol`, all async):
  - `doc_facts(title: str, text: str, other_titles: dict[str, str]) -> DocFacts` where `DocFacts(date: date | None, replaces: list[str])` (`replaces` = doc ids picked from `other_titles` `{id: title}`).
  - `extract_claims(question: str, doc: DocumentInfo, text: str) -> ExtractedClaims` where `ExtractedClaims(claims: list[ExtractedClaim], dropped_non_factual: int)` and `ExtractedClaim(text: str, quote: str, important: bool)`.
  - `group_claims(question: str, claims: list[Claim]) -> dict[str, str]` (claim id → short subject).
  - `judge(claim: Claim, passages: list[Passage]) -> Judgment`
  - `reword(claim: Claim, tried: list[str]) -> str`
  - `write_answer(question: str, claims: list[Claim], verdicts: list[Verdict], caveat: str | None) -> str` — 1–3 sentences, uses only the given claims.
- `ClaudeReasoner(model: str)` implements it with `ChatAnthropic(model=..., temperature=0).with_structured_output(<pydantic model>)` per call; prompts in `prompts.py`. Every passage and document text is wrapped in tags and the system prompt says text inside them is data, not instructions.
- `FakeReasoner` takes scripted return values per method (dicts keyed by claim id / doc id) and records calls.

- [ ] **Step 1: Failing test** `test_claude_reasoner_builds_structured_calls`: monkeypatch `ChatAnthropic` with a stub whose `with_structured_output(schema)` returns an object whose `ainvoke` returns a fixed instance of `schema` → `judge(...)` returns that `Judgment`; the stub saw the passage text inside `<passage id="...">` tags. No network.
- [ ] **Step 2:** Run → FAIL. Implement. Run → PASS. Commit "verifier: reasoner interface and Claude implementation".

### Task 6: The graph

**Files:**
- Create: `verifier/graph.py`
- Test: `verifier/tests/test_graph.py`

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces: `build_graph(reasoner: Reasoner, search: EvidenceSearch, *, checkpointer=None) -> CompiledStateGraph`.
- State (`TypedDict`): `question`, `documents: dict[str, DocumentInfo]`, `texts: dict[str, str]`, `claims: Annotated[list[Claim], operator.add]`, `dropped_non_factual: Annotated[int, operator.add]`, `checks: Annotated[list[ClaimCheck], operator.add]`, `verdicts: dict[str, Verdict]`, `pending: list[PersonQuestion]`, `person_answers: dict[str, str]`, `report: Report | None`.
- Nodes and edges:
  - `doc_facts` (one call per doc; manifest date wins; a text date sets `date_source="text"`) → fan out `Send("extract", {...doc})` per doc.
  - `extract` → appends claims (ids `doc#1..n`). → `group` (sets `subject`) → fan out `Send("verify", {"claim": c})` per claim.
  - `verify` (one claim): query 1 = claim text, query 2 = subject; merge, dedupe by passage id; `judge`; while `needs_recheck` and tries < 3: `reword` → search → judge. Any exception from search/judge → `Judgment(stances=[], states_claim=True, confidence="low", reason="check failed: <ExcType>")` and stop retrying. Appends one `ClaimCheck`.
  - `label` → runs `label_claim` for every claim, fills `verdicts` and `pending` (deduped by question id).
  - `ask_person` → if `pending`: `answer = interrupt(question.model_dump())`, store in `person_answers`, apply it (`low_confidence`: "true" → keep the label, `settled_by_person=True`; "false" → label `contradicted`, `settled_by_person=True`, reason "a person said this is false"; "can't tell" → unchanged; `replaces_vs_date`: chosen doc wins → re-label the claims of that pair as `outdated`/`single_source`, settled; "neither" → stays `contradicted`, settled=False; `guessed_date`: "yes" keeps date, "no" sets `date=None` and re-runs `label` for that doc's claims), pop it, loop back to `ask_person`; else → `report`.
  - `report` → `choose_answer`; if answered, `write_answer`; builds `Report`.
- [ ] **Step 1: Failing tests** with `FakeReasoner` + `EvidenceSearch` over `MemoryVectorStore`, `MemorySaver` checkpointer, a 3-doc leave scenario (old 20 days, new 25 days replacing old, FAQ undated):
  - `test_every_claim_gets_a_verdict`: `len(report.verdicts) == len(report.claims)`.
  - `test_recheck_stops_after_two_extra_tries`: judge always low → `check.tries == 3`, `reword` called twice.
  - `test_failed_judge_does_not_drop_claim`: judge raises for one claim → that claim has a verdict, reason contains "check failed".
  - `test_interrupt_and_resume`: scenario with a `replaces_vs_date` clash → first `ainvoke` stops with `__interrupt__`; `ainvoke(Command(resume="Leave Policy 2025"), config)` finishes; `report.person_answers` has it and the old claim is `outdated` with `settled_by_person=True`.
  - `test_end_to_end_answer`: answer status `"answered"`, `claim_ids` points at the 25-day claim, caveat mentions it replaced the 2023 figure.
- [ ] **Step 2:** Run → FAIL. Implement. Run → PASS. Commit "verifier: LangGraph graph with fan-out, re-check and person questions".

### Task 7: Report writer

**Files:**
- Create: `verifier/report.py`
- Test: `verifier/tests/test_report.py`

**Interfaces:**
- Produces: `render_markdown(report: Report) -> str`; `write_report(report: Report, folder: Path) -> tuple[Path, Path]` (writes `report.md`, `report.json` = `report.model_dump_json(indent=2)`).
- Markdown order (spec): `## Answer` (answer text, or "Not enough evidence to answer." + why; caveat in italics); `## Claims` table: Claim | Label | Source (doc title, date) | Evidence (≤200-char quotes with doc title and page/heading); `## Contradictions` (each `contradicted`/`outdated` claim with both sides); `## Settled by a person` (question → answer); footer line with counts per label and `dropped_non_factual`.
- [ ] **Step 1: Failing tests:** `test_answer_first`, `test_not_enough_evidence_wording`, `test_every_claim_in_table`, `test_contradiction_shows_both_sides`, `test_json_round_trips` (`Report.model_validate_json` equals original).
- [ ] **Step 2:** Run → FAIL. Implement. Run → PASS. Commit "verifier: Markdown and JSON report".

### Task 8: Command line

**Files:**
- Create: `verifier/__main__.py`, `verifier/README.md` (how to start the DB container, env vars, run)
- Test: `verifier/tests/test_cli.py`

**Interfaces:**
- `python -m verifier run <folder> [--answers answers.yaml]`. Flow: `load_manifest` → `build_store` → `ensure_schema` → `index_documents` → `build_graph(ClaudeReasoner(settings.model), EvidenceSearch(...), checkpointer=MemorySaver())` → loop: `ainvoke`; on `__interrupt__`, take the answer from `answers.yaml` (`{question_id: answer}`) if present, else print the question text + numbered choices and read stdin; resume with `Command(resume=...)` → `write_report` → print the two paths.
- `main(argv: list[str], *, reasoner=None, store=None, input_fn=input) -> int` so tests inject fakes.
- README includes the DB command: `docker run -d --name claim-verifier-db --restart no -e POSTGRES_PASSWORD=postgres -p 55432:5432 public.ecr.aws/supabase/postgres:17.6.1.165` and removal `docker rm -f claim-verifier-db`.
- [ ] **Step 1: Failing tests:** `test_cli_uses_answers_file` (fakes, a clash scenario, answers file given → no `input_fn` call, report files exist); `test_cli_prompts_when_no_answer` (`input_fn` returns `"1"` → first choice used).
- [ ] **Step 2:** Run → FAIL. Implement. Run → PASS. Commit "verifier: command line".

### Task 9: Test set and acceptance run

**Files:**
- Create: `verifier/testsets/leave-policy/` — `manifest.yaml`, 5 made-up `.md` documents (fictional org "Riverbend Learning Trust"), `key.yaml`, `answers.yaml`; `verifier/testsets/no-answer/manifest.yaml` (same docs, question about something absent, e.g. maternity leave); `verifier/acceptance.py`
- Test: `verifier/tests/test_acceptance.py`

**Interfaces:**
- `key.yaml`: `answer_status`, and a list of `{doc, contains, label}` — a claim from `doc` whose text contains `contains` (case-insensitive) must get `label`.
- `check(report: Report, key: dict) -> list[str]` returns mismatch lines (empty = pass); `python -m verifier.acceptance <folder>` prints them, exits 1 on any.
- The documents must contain (spec success test): one outdated claim (2023 policy: 20 days; 2025 policy: 25 days, states it replaces the 2023 policy); one plain contradiction (two same-dated documents disagreeing on carry-over days); one unsupported claim (only one document mentions a "wellness day"); one claim verified by two documents (both policies and the FAQ say leave requests go to the line manager); one dates-vs-replaces clash (a memo dated 2022 claiming to replace the 2025 policy on sick leave). `answers.yaml` settles the clash.
- [ ] **Step 1: Failing test** `test_check_reports_mismatch` / `test_check_passes` with hand-built `Report`s. Run → FAIL. Implement `acceptance.py`. Run → PASS. Commit "verifier: test set and acceptance check".
- [ ] **Step 2 (after Ollama + bge-m3 are installed, after 8 pm):** `curl -s localhost:11434/api/tags` lists `bge-m3`; start the DB container (README); `ANTHROPIC_API_KEY` set in the shell by the owner; run `python -m verifier run verifier/testsets/leave-policy --answers verifier/testsets/leave-policy/answers.yaml` then `python -m verifier.acceptance verifier/testsets/leave-policy`. Expected: no mismatch lines. Same for `no-answer` (expected `answer_status: not_enough_evidence`).
- [ ] **Step 3:** If mismatches: adjust prompts in `verifier/prompts.py` only, at most 5 runs in total. Read `report.md` yourself. Any label still wrong after that is written into `verifier/README.md` under "Known misses", not hidden.
- [ ] **Step 4:** `docker rm -f claim-verifier-db`. Commit prompts + README — "verifier: tuned prompts, acceptance result <date from `date`>".

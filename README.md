# Claim Verifier

**Give it a question and a pile of documents on the same topic. It tells you what the documents actually support,
what is outdated, where they disagree, and when there is simply not enough evidence to answer — with the passage
behind every verdict.**

AI answers sound convincing even when the information behind them is incomplete, outdated, contradictory, or not
supported at all. Searching documents (RAG) finds passages, but finding a passage does not make the answer right.
The Claim Verifier checks the documents against each other before anyone trusts the answer.

- [What it does](#what-it-does)
- [Where you can use it](#where-you-can-use-it)
- [Where you should not use it](#where-you-should-not-use-it)
- [What you need](#what-you-need)
- [Setup (once per machine)](#setup-once-per-machine)
- [Using it in your browser](#using-it-in-your-browser)
- [Using it from the terminal, step by step](#using-it-from-the-terminal-step-by-step)
- [Reading the report](#reading-the-report)
- [Answering questions in advance and unattended runs](#answering-questions-in-advance-and-unattended-runs)
- [Settings](#settings)
- [Checking that it still works](#checking-that-it-still-works)
- [Troubleshooting](#troubleshooting)
- [How it works inside](#how-it-works-inside)
- [Limits and known misses](#limits-and-known-misses)
- [Project layout and credits](#project-layout-and-credits)

---

## What it does

For one question and a folder of documents, it:

1. **reads every document**, finds its date and whether it says it replaces another document;
2. **pulls out every factual claim** on the topic (with the exact sentence it came from);
3. **searches for evidence** for each claim across all the documents (hybrid meaning + keyword search);
4. **labels each claim**:

   | Label | Meaning |
   |---|---|
   | **Verified** | Another document says the same thing, and nothing newer disagrees. |
   | **Only one document says this** | Its own document states it, no other document confirms it, nothing newer disagrees. |
   | **Documents disagree** | Documents of the same (or unknown) age say different things and no rule can decide. |
   | **Outdated** | A newer document — or one that says it replaces this one — says something different. |
   | **No evidence** | The claim cannot be traced to any passage (or could not be checked). |

5. **re-checks claims it is unsure about** with a reworded search (up to 2 extra tries);
6. **asks you** when something important cannot be decided by rule (see [questions it may ask](#questions-it-may-ask));
7. **writes a report** (`report.md` for people, `report.json` for programs) that answers the question first —
   with a visible caveat when the answer rests on thin evidence — or says **"Not enough evidence to answer"**.

How "outdated" is decided: the newer document wins by date. A document that explicitly says it replaces another
also wins. When the two rules point different ways (an *older* memo says it replaces a *newer* policy), it does not
guess — it asks you.

When it answers: it gives the best-supported answer with a caveat ("Only one document states this. It replaced the
2023 figure."). It declines only when the claim that would answer the question has no evidence, or its disagreement
was not settled.

## Where you can use it

Anywhere a set of documents on one subject has grown over time and you need an answer you can defend.

| Situation | Example question | What it catches |
|---|---|---|
| **Policies and handbooks** (HR, finance, IT) | "How many days of annual leave does a full-time employee get?" | The 2023 policy still floating around, an FAQ that disagrees with the handbook. |
| **Checking an AI or RAG answer** before it goes out | "What is the refund window?" | Answers built on one stale passage; claims no document supports. |
| **Curriculum and teaching material** | "Which chapters are in the Grade 5 science syllabus this year?" | Old syllabus versions, guides that contradict the notified scheme of studies. |
| **Programme and project documents** | "What is the target number of schools in phase 2?" | Proposal vs. revised work plan vs. progress report giving different numbers. |
| **Product docs and release notes** | "Does the app work offline?" | Docs written for an old version; a release note that quietly changed behaviour. |
| **SOPs and operating procedures** | "Who approves a field visit?" | Two SOPs with different approvers, a memo that overrides one of them. |
| **Research and briefing notes** | "What was the pilot's attendance rate?" | Figures quoted differently across reports; numbers with no source. |
| **Contracts and terms (made-up or public)** | "What is the notice period?" | An amendment replacing a clause; a summary sheet that got it wrong. |
| **Before publishing a knowledge base** | Any "single source of truth" question | Finds which pages need updating or retiring. |

It is most useful when:

- there are **2–15 documents** on **one topic**, some newer than others;
- the question has a **factual answer** (a number, a rule, a name, a date), not an opinion;
- you can put a **date** on each document (or the document states its own date).

## Where you should not use it

- **Confidential documents with the Claude option.** With *Claude*, document text is sent to Anthropic. For
  confidential documents use *On this computer* (the default): nothing leaves the machine. See
  [Local or Claude](#local-or-claude).
- **Opinion or judgement questions** ("Is this policy fair?"). It checks facts, not views.
- **Scanned PDFs and images.** There is no OCR: a PDF made of page pictures is refused. Use the original Word or text file.
- **Urdu / Roman Urdu documents.** Not tuned or tested for them yet.
- **Hundreds of documents.** Every claim is checked; large sets get slow and use a lot of model calls.
- **As a final authority.** It shows you the evidence so a person can decide faster. Read the report.

## File types

| Type | Extensions | How it is read |
|---|---|---|
| Text | `.md` `.txt` `.csv` | as it is (UTF-8; Windows-1252 also accepted) |
| PDF | `.pdf` | pdfplumber, the same reader as Ask the Workspace (scanned PDFs are refused: no OCR) |
| Word, Excel, PowerPoint | `.docx` `.xlsx` `.pptx` | markitdown, as in Ask the Workspace; tables stay tables, Excel rows keep their column names |
| Older and open formats | `.doc` `.rtf` `.odt` `.xls` `.ods` `.ppt` `.odp` | converted by LibreOffice first (must be installed), then read as above |

Up to 25 MB per file (100 MB per check in the browser). Password-protected or damaged files are refused with a
sentence saying what to do. Images (`.png`, `.jpg`) are not read.

## Local or Claude

The reading, searching and storing always happen on this computer. Only the reasoning steps (pulling out claims,
judging evidence, writing the answer) use a language model, and you choose which one for each check:

| | On this computer (`local`, default) | Claude (`claude`) |
|---|---|---|
| Document text leaves the computer | **no** | yes, to Anthropic |
| Model | `qwen3.5:4b` in Ollama, as Ask the Workspace uses | `claude-sonnet-5-5` |
| Time for 5 short documents | minutes (no graphics card: each call is slow) | about 1 minute |
| Accuracy | lower: a small model misses more | higher |

Download the local model once (3.4 GB):

```bash
docker exec claim-verifier-ollama ollama pull qwen3.5:4b
```

In the terminal: `uv run python -m verifier run my-check --backend local` (or `--backend claude`). In the page:
step 4. Until the local model is downloaded, *On this computer* shows "Not ready" and only Claude can run.

## What you need

| Thing | Why | Check |
|---|---|---|
| Linux or macOS with **Docker** | runs the document database and the search model | `docker ps` |
| **uv** (Python package manager) | installs and runs the project (Python 3.12+) | `uv --version` |
| **Claude Code**, installed and signed in (only for the Claude option) | Claude is reached through your Claude Code login, no API key needed | `claude --version` |
| **qwen3.5:4b** in Ollama (only for the local option) | the private reasoning model, 3.4 GB | `docker exec claim-verifier-ollama ollama list` |
| **LibreOffice** (optional) | reads older/open formats: .doc .rtf .odt .xls .ods .ppt .odp | `soffice --version` |
| ~2 GB disk, ~3 GB free RAM | database + search model (bge-m3) | |

Cost and time, measured on 5 short documents: about **1 minute** per run, about **half a cent per model call**
(a run makes a few dozen calls). Calls count against your Claude plan's usage.

## Setup (once per machine)

```bash
git clone git@github.com:Nimra26/claim-verifier.git
cd claim-verifier
uv sync                                   # installs Python packages into .venv
```

Start the two helper containers. Neither starts by itself after a reboot (`--restart no`):

```bash
# 1. Document database (Postgres with pgvector)
docker run -d --name claim-verifier-db --restart no \
  -e POSTGRES_PASSWORD=postgres -p 127.0.0.1:55432:5432 \
  public.ecr.aws/supabase/postgres:17.6.1.165

# 2. Search model server (Ollama); the volume keeps the model between runs
docker run -d --name claim-verifier-ollama --restart no \
  -p 127.0.0.1:11434:11434 -v claim-verifier-ollama:/root/.ollama \
  ollama/ollama:latest

# 3. Download the search model (first time only, ~1.2 GB)
docker exec claim-verifier-ollama ollama pull bge-m3
```

Try it on the included example — in your browser:

```bash
uv run python -m verifier.web          # opens http://127.0.0.1:8765, then click "Try the example" → "Run check"
```

or in the terminal:

```bash
uv run python -m verifier run verifier/testsets/leave-policy
```

When you are done for the day:

```bash
docker rm -f claim-verifier-db claim-verifier-ollama   # the model volume stays
```

Next time, run only steps 1 and 2 (no download).

## Using it in your browser

```bash
uv run python -m verifier.web
```

This opens **http://127.0.0.1:8765** (only this computer can reach it; `--port 9000` to change the port,
`--no-browser` to not open a tab). Stop it with Ctrl+C.

1. **Check the three dots at the top**: Database, Search model, and the reasoning model you picked in step 4. Green means ready; hover a red one to see
   what to start.
2. **Type your question.**
3. **Add documents**: drop files on the box or click *choose files* (PDF, Word, Excel, PowerPoint, OpenDocument, RTF,
   Markdown, CSV or plain text; up to 25 MB each — see [File types](#file-types)). For each one you can set the
   **date** it was issued (filled in automatically when the file name or first lines contain a date like
   `2025-03-01`) and tick which other documents it **replaces**. Remove one with ×.
4. **Choose where the reasoning runs**: *On this computer (private)* or *Claude (cloud)*. The page remembers your
   choice; each report says which model checked it.
   No documents at hand? Click **Try the example** to load the five Riverbend test documents.
4. Click **Run check**. The steps show *Reading documents → Checking claims → Your decision → Report*
   (about a minute for a few short documents).
5. If it needs a decision, a blue box asks the question with one button per choice. The check waits until you click
   one (or **Stop the check**).
6. **Read the report**: the answer on top (with its caveat in yellow, or *Not enough evidence to answer*), the
   claims with coloured labels — click a label at the top to show only those — each claim's reason and
   **Evidence** (click to open), the contradictions side by side, and what you settled.
   **Download report (.md)** and **Download data (.json)** save it; **Change and run again** keeps your documents.

One check runs at a time. Each check is saved in `~/.claim-verifier/runs/<date-time-id>/` (its documents,
`manifest.yaml`, `report.md`, `report.json`); set `VERIFIER_RUNS_DIR` to save elsewhere. That folder can also be
re-run from the terminal: `uv run python -m verifier run ~/.claim-verifier/runs/<id>`.

## Using it from the terminal, step by step

### 1. Make a folder with your documents

```
my-check/
├── manifest.yaml
├── policy-2023.md
├── policy-2025.md
└── faq.md
```

- Files can be any type listed in [File types](#file-types).
- For `.md` / `.txt`, start each file with a `# Title` line — the report uses it to name the document. Other files
  are named by their first `# ` heading if they have one, otherwise by the file name (`Leave Policy 2025.pdf` → `Leave Policy 2025`).
- The file name without extension becomes the document's id (`policy-2025`).

### 2. Write `manifest.yaml`

```yaml
question: How many days of annual leave does a full-time employee get?
documents:
  - file: policy-2023.md
    date: 2023-01-15
  - file: policy-2025.md
    date: 2025-03-01
    replaces: [policy-2023.md]     # optional: files this one replaces
  - file: faq.md                   # no date: it will try to read one from the text
```

| Field | Required | Notes |
|---|---|---|
| `question` | yes | One factual question. The report answers this. |
| `documents[].file` | yes | File name inside the folder. |
| `documents[].date` | no | `YYYY-MM-DD`, when the document was issued. If missing, the date is read from the text and you are asked to confirm it; with no date at all, the document can never make another one "outdated". |
| `documents[].replaces` | no | List of file names this document replaces. A document's own text saying "this replaces…" is also detected. |

A full example is in [`verifier/testsets/leave-policy/`](verifier/testsets/leave-policy/).

### 3. Run it

```bash
uv run python -m verifier run my-check
```

You will see:

```
Indexing 3 documents...
Checking claims...
Report written: my-check/report.md
Report written: my-check/report.json
```

### 4. Answer its questions (if any)

#### Questions it may ask

| When | Example | Choices |
|---|---|---|
| A document says it replaces a **newer** one | "Sick Leave Memo 2022 says it replaces Leave Policy 2025, but Leave Policy 2025 has a newer date. Which one is current?" | either document, or `neither` |
| It is still unsure about an **important** claim after re-checking | "We are not sure about this: … Is it true?" | `true`, `false`, `can't tell` |
| It **guessed** a document's date from its text | "We guessed the date of Leave FAQ from its text. Is 2025-06-10 right?" | `yes`, `no` |

Type the number or the exact choice. After 3 invalid tries it uses the last (non-committal) choice and says so.
Every question and your answer appear in the report under "Settled by a person".

### 5. Read `report.md`

See the next section.

## Reading the report

`report.md` has these parts, in order:

1. **Answer** — the answer in 1–3 sentences, with an italic caveat under it when the evidence is thin, or
   *"Not enough evidence to answer."* and why.
2. **Claims** — one row per claim: the claim, its label, **why** it got that label, its source document and date,
   and the evidence passage(s). Low-confidence judgements are marked.
3. **Contradictions** — every outdated or disagreeing claim with **both sides** quoted and both sources named.
4. **Settled by a person** — each question asked and the answer given.
5. **Could not check** — anything that failed (a model call, a document). Never hidden.
6. A footer with the count of each label.

Real excerpt from the example set:

```markdown
## Answer

A full-time employee gets 25 days of annual leave per year. Only one document, Leave Policy 2025,
states this figure, and it replaced the figure in Leave Policy 2023.

*Only one document states this. It replaced Leave Policy 2023's figure.*

| Claim | Label | Why |
| A full-time employee gets 20 days of annual leave per year. | Outdated | Leave Policy 2025 is newer or replaces Leave Policy 2023 and says something different. |
| Leave requests go to the employee's line manager. | Verified | Leave FAQ, Leave Policy 2025 also states this. |
| An employee can carry over up to 5 days ... | Documents disagree | Staff Handbook Extract says something different and we cannot tell which is current. |
```

`report.json` holds the same data (documents, claims, checks, verdicts, answers, failures) for other programs.

## Answering questions in advance and unattended runs

To answer in advance, write an answers file mapping question id → choice:

```yaml
# answers.yaml
"replaces:leave-policy-2025|sick-leave-memo-2022": "Leave Policy 2025"
"confidence:leave-faq#2": "true"
"date:leave-faq": "yes"
```

| Question id | Form |
|---|---|
| replaces vs. date | `replaces:<doc-a>\|<doc-b>` (the two document ids, sorted alphabetically) |
| unsure claim | `confidence:<claim-id>` (claim ids are shown in `report.json`) |
| guessed date | `date:<doc-id>` |

```bash
uv run python -m verifier run my-check --answers my-check/answers.yaml
```

An answer that is not one of the offered choices is ignored and the question is asked instead.

For runs nobody watches (scripts, repeat checks), add `--unattended`: unanswered questions get the non-committal
choice (`neither`, `can't tell`, or `no` for a guessed date) and each is printed as
`Unattended: <id> ... answered with: <choice>`.

**Exit codes** (useful in scripts):

| Code | Meaning |
|---|---|
| 0 | Done, report written. |
| 2 | Bad input: missing folder / manifest / file, unsupported file type, Claude Code not found, or the report could not be written. |
| 3 | Database or Ollama not reachable — start the containers. |
| 4 | The check could not be completed (model calls failed, no claims read). The report says so and lists the failures. |

## Settings

All optional, set as environment variables:

| Variable | Default | What it does |
|---|---|---|
| `VERIFIER_BACKEND` | `local` | `local`: a model in Ollama on this computer, nothing leaves it. `cli` (or `claude`): Claude through the Claude Code login (no key). `api`: the Anthropic API, needs `ANTHROPIC_API_KEY`. The page and `--backend` override it per check. |
| `VERIFIER_LOCAL_MODEL` | `qwen3.5:4b` | The Ollama model for `local` (the same one Ask the Workspace answers with). |
| `VERIFIER_MODEL` | `claude-sonnet-5-5` | Which Claude model reasons. |
| `VERIFIER_DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@localhost:55432/postgres` | Document database. |
| `VERIFIER_OLLAMA_URL` | `http://localhost:11434` | Search model server. |
| `VERIFIER_EMBED_MODEL` | `bge-m3` | Search (embedding) model. |
| `VERIFIER_TOP_K` | `8` | Passages fetched per search. |
| `VERIFIER_CHUNK_SIZE` / `VERIFIER_CHUNK_OVERLAP` | `800` / `120` | How documents are split for search. |

With the `cli` backend each model call runs `claude -p` with no tools, plugins, MCP servers or settings files, from
an empty temporary folder, at most 4 at a time; any `ANTHROPIC_API_KEY` in your shell is removed for that call so a
stale key cannot override your login.

## Checking that it still works

```bash
# Unit tests (offline, no containers, no model): 129 tests
uv run pytest -q -p no:cacheprovider verifier/tests

# One tiny live call per request type (a few cents)
uv run python -m verifier.probe_model

# Full acceptance check against a known answer key (containers must be running)
uv run python -m verifier run verifier/testsets/leave-policy \
  --answers verifier/testsets/leave-policy/answers.yaml --unattended
uv run python -m verifier.acceptance verifier/testsets/leave-policy     # expect: No mismatches.

uv run python -m verifier run verifier/testsets/no-answer --unattended
uv run python -m verifier.acceptance verifier/testsets/no-answer        # expect: No mismatches.
```

The test sets use a made-up organisation (Riverbend Learning Trust) with one outdated policy, a plain
contradiction, a claim confirmed by two documents, single-source claims, and an older memo that claims to replace a
newer policy. `no-answer` asks something none of the documents cover.

Last live result (27 Nov 2026, claude-sonnet-5-5 via Claude Code): both test sets pass with no mismatches.

## Troubleshooting

| You see | Do this |
|---|---|
| `Could not reach the document database` / exit 3 | Start `claim-verifier-db` (setup step 1). Check `docker ps`. |
| Ollama connection error / exit 3 | Start `claim-verifier-ollama` (setup step 2); make sure `bge-m3` was pulled. |
| `The Claude Code command (claude) was not found.` | Install Claude Code and sign in (`claude` once interactively). |
| `Claude Code returned no …: Not logged in` in "Could not check" | Run `claude` and sign in, then run again. |
| `unsupported file type` | See [File types](#file-types); save the file as PDF or Word and add that. |
| `Could not read a document. … is a scan` | The PDF is pictures of pages; use the original Word or text file. |
| `… needs LibreOffice to read it` | Install LibreOffice (`sudo apt install libreoffice`), or save the file as .docx / .xlsx / .pptx. |
| `manifest names a file that does not exist` | Check the file names in `manifest.yaml` match exactly. |
| Page says “Another check is still running” | Wait for it, or answer/stop its question; restarting `verifier.web` also clears it. |
| Port 55432 or 11434 already in use | Another container is running; stop it, or use different ports and set `VERIFIER_DATABASE_URL` / `VERIFIER_OLLAMA_URL`. |
| Everything is "No evidence" | Look at "Could not check" in the report — usually model calls failed. |

## How it works inside

A fixed [LangGraph](https://github.com/langchain-ai/langgraph) graph — every claim takes the same path, so none is
skipped:

```
doc_facts (date, "replaces")   one per document, in parallel
   └▶ extract claims            one per document, in parallel
        └▶ group claims         same-subject claims across documents
             └▶ verify          one per claim, in parallel:
                  search ▶ judge ▶ (unsure? reword + search again, ≤2 times)
                  └▶ label      rules in plain Python: dates, "replaces", support, contradiction
                       ↕ ask_person   pauses the graph (LangGraph interrupt) for your answer, then re-labels
                       └▶ report      answer + claims + contradictions + settled + failures
```

- **Search** is Laraib's RAG pipeline from Ask the Workspace: hybrid pgvector + keyword search over chunks embedded
  with bge-m3 through Ollama.
- **Reasoning** (reading, extracting, judging, rewording, writing the answer) is the model you choose (local
  Ollama or Claude) with structured output, so every reply is checked against a schema.
- **Labelling is not left to the model.** The model says which passages support or contradict a claim; plain
  Python rules turn that plus the dates into the label. A claim's own sentence never counts as its own support.
- **Document text is treated as data**, wrapped in tags and escaped, so instructions hidden in a document are not
  followed.

Full design and decisions: [docs/design.md](docs/design.md). Build plan: [docs/plan.md](docs/plan.md).

## Limits and known misses

- Evidence from another document shows the start of the matching chunk, not the exact matching sentence. With short
  documents the quote starts at the title. Labels are unaffected.
- No OCR: scanned PDFs are refused. Evidence names the document and passage, not the page number.
- One run, one question: answers are not remembered between runs (use an answers file, or the browser page).
- The browser page forgets running checks when it is restarted (finished reports stay in the runs folder).
- Documents with the same file name in different folders share an id; re-running replaces the earlier copy in the
  database. Run one folder at a time.
- English only so far.

## Project layout and credits

```
verifier/            the Claim Verifier
  __main__.py        command line: python -m verifier run <folder>
  web.py             browser page: python -m verifier.web
  webui/index.html   the page itself, in the Ask the Workspace look (no outside scripts; works offline)
  webui/static/      fonts, logo and header theme copied from Ask the Workspace
  graph.py           the LangGraph graph
  rules.py           labelling, outdated rule, when to ask a person, which claim answers
  reasoner.py        model calls (local Ollama, Claude Code login or API) and their schemas
  prompts.py         all prompt text
  search.py          evidence search over the indexed documents
  store.py           database setup and indexing
  manifest.py        reads manifest.yaml and the documents
  documents.py       turns PDF / Word / Excel / PowerPoint / text files into text
  report.py          writes report.md / report.json
  acceptance.py      compares a report with an answer key
  probe_model.py     one live call per request type
  tests/             129 offline tests (fake model, fake search)
  testsets/          leave-policy and no-answer example sets with answer keys
src/                 retrieval code copied from Ask the Workspace (see NOTICE.md)
docs/                design and build plan
```

Retrieval: Laraib's RAG pipeline from **Ask the Workspace**, copied into `src/` unchanged — see
[NOTICE.md](NOTICE.md). Licence: MIT ([LICENSE](LICENSE)).

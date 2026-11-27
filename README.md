# Claim verifier

Reads a folder of `.md`/`.txt` documents plus a `manifest.yaml`, checks the claims in them against each other, and writes `report.md` and `report.json` into that folder.

Given several documents on one topic (some correct, some outdated, some contradicting each other), it pulls out each
factual claim, searches for evidence for every claim, labels it (verified, single source, contradicted, outdated, no
evidence), re-checks uncertain ones, asks a person to settle important ambiguities, and says so when there is not
enough evidence to answer. Built as a fixed LangGraph graph; design in [docs/design.md](docs/design.md).

Retrieval is Laraib's RAG pipeline from Ask the Workspace, copied into `src/` (see [NOTICE.md](NOTICE.md)).

## Quick start

```
uv sync
docker run -d --name claim-verifier-db --restart no -e POSTGRES_PASSWORD=postgres -p 127.0.0.1:55432:5432 public.ecr.aws/supabase/postgres:17.6.1.165
docker run -d --name claim-verifier-ollama --restart no -p 127.0.0.1:11434:11434 -v claim-verifier-ollama:/root/.ollama ollama/ollama:latest
docker exec claim-verifier-ollama ollama pull bge-m3      # first time only
uv run python -m verifier run verifier/testsets/leave-policy
docker rm -f claim-verifier-db claim-verifier-ollama       # when finished
```

Claude is reached through the local Claude Code login (`claude` on PATH); no API key needed.

## Folder layout

```yaml
# manifest.yaml
question: How many days of annual leave does a full-time employee get?
documents:
  - file: leave-2023.md
    date: 2023-01-15        # optional
  - file: leave-2025.md
    date: 2025-03-01
    replaces: [leave-2023.md]   # optional
```

## One-time setup

Start the document database (Postgres with pgvector):

```
docker run -d --name claim-verifier-db --restart no -e POSTGRES_PASSWORD=postgres -p 127.0.0.1:55432:5432 public.ecr.aws/supabase/postgres:17.6.1.165
```

Remove it when finished:

```
docker rm -f claim-verifier-db
```

Start Ollama for the embedding model (the volume keeps bge-m3, so it downloads only the first time):

```
docker run -d --name claim-verifier-ollama --restart no -p 127.0.0.1:11434:11434 -v claim-verifier-ollama:/root/.ollama ollama/ollama:latest
docker exec claim-verifier-ollama ollama pull bge-m3
```

Remove both containers when finished (the volume stays): `docker rm -f claim-verifier-db claim-verifier-ollama`

## Environment variables

- `VERIFIER_BACKEND` (default `cli`): how Claude is reached. `cli` runs the local Claude Code command (`claude -p`)
  with its login, so no API key is needed; each call runs with no tools, plugins, MCP servers or settings files, from an
  empty temporary folder, at most 4 at a time. `api` uses the `anthropic` SDK directly.
- `ANTHROPIC_API_KEY`: needed only with `VERIFIER_BACKEND=api`. The `cli` backend removes it from the command's
  environment so a stale key cannot override the login.
- `VERIFIER_DATABASE_URL` (default `postgresql+asyncpg://postgres:postgres@localhost:55432/postgres`)
- `VERIFIER_OLLAMA_URL` (default `http://localhost:11434`)
- `VERIFIER_EMBED_MODEL` (default `bge-m3`)
- `VERIFIER_MODEL` (default `claude-sonnet-5-5`)
- `VERIFIER_TOP_K`, `VERIFIER_CHUNK_SIZE`, `VERIFIER_CHUNK_OVERLAP`

The retrieval code's own app settings are never used. Importing it builds those settings, so the verifier
(`verifier/appenv.py`, imported first) sets dummy values for the app's secret variables (`ACCESS_TOKEN_SECRET_KEY`,
`REFRESH_TOKEN_SECRET_KEY`, `RESET_PASSWORD_TOKEN_SECRET`, `VERIFICATION_TOKEN_SECRET`, `DATABASE_URL`) when they are
not already set, and points `SETTINGS_ENV_FILE` at a file that does not exist, so the app's `.env` is never read.

## Run

```
uv run python -m verifier run <folder>
```

When the checker needs a person to decide something, it prints the question with numbered choices; type the number or the exact choice text. After 3 invalid tries it uses the last choice and says so.

To answer in advance, pass `--answers answers.yaml`, a map of question id to one of the offered choices:

```yaml
"replaces:leave-2023|leave-2025": "Leave Policy 2025"
```

A question id names the two documents in sorted order (`replaces:<a>|<b>`). An answer that is not among the choices
is ignored and the question is asked instead.

For a repeatable run that never waits for a person, add `--unattended`: any question not answered in the answers file
gets its last, non-committal choice ("neither", "can't tell"; "no" for a guessed date, which leaves the document
undated), and each such question is printed as `Unattended: <id> ... answered with: <choice>`.

Exit codes: 0 done, 2 bad input (missing folder, manifest or API key) or the report could not be written, 3 database or
Ollama unreachable, 4 the check could not be completed (model calls failed and no claims were read; the report says so
and lists the failures).

## Acceptance check

Run a test set unattended, then compare its report with the set's answer key (`key.yaml`):

```
uv run python -m verifier run verifier/testsets/leave-policy --answers verifier/testsets/leave-policy/answers.yaml --unattended
uv run python -m verifier.acceptance verifier/testsets/leave-policy
```

Any failure recorded in the report counts as a mismatch.

To check that the model accepts every structured-output request (one tiny call per schema, a few cents):

```
uv run python -m verifier.probe_model
```

## Tests

```
uv run pytest -q -p no:cacheprovider verifier/tests
```

## Live results (27 Nov 2026, claude-sonnet-5-5 via the Claude Code login)

- `probe_model`: all six output schemas returned valid structured output.
- `leave-policy`: every key claim got its expected label, answer "25 days" with the single-source caveat, and the
  scripted dates-vs-"replaces" question was asked and settled. `no-answer`: "Not enough evidence to answer."
- Prompt tuning took 1 change (run 1 missed 6 key claims because extraction kept only claims answering the
  question; it now extracts every claim on the topic and skips statements about the document itself). 3 live runs.

## Known misses

- Evidence from another document shows the start of the matching chunk, not the matching sentence. The test
  documents are short, so each is one chunk and the quote starts at its title. Labels are unaffected.

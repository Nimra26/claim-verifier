# Where the code in `src/` comes from

`src/` is a copy of the retrieval part of **Ask the Workspace** (Laraib's RAG pipeline), taken from branch
`claim-verifier` at commit `f34a275` on 27 Nov 2026. Only the 43 files the verifier loads were copied, unchanged:
hybrid pgvector + keyword search (`src/infrastructure/vector_db/pgvector_store.py`), the retrieval and ingestion
services (`src/rag/pipeline/services.py`), chunking (`src/rag/ingestion/chunking/`), the Ollama/OpenAI-compatible
embedding client (`src/infrastructure/llm/openai_embeddings.py`) and the settings they import.

It is a copy, so fixes made in Ask the Workspace do not reach it automatically. Ask the Workspace is built on an
MIT-licensed open-source base; its licence is kept in `LICENSE`.

## The browser page's look

`verifier/webui/static/` holds the Ask the Workspace page assets, copied unchanged so the Claim Verifier page looks
like the app: the self-hosted fonts (Fredoka, Poppins, Noto Nastaliq Urdu; SIL Open Font Licence, see
`fonts/LICENSE`), the header theme `css/app_theme.css`, the Taleemabad wordmark and the favicon. The page's colours,
cards, question bubble, answer card and source cards follow `templates/workspace_ui.html`.

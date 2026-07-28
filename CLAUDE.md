# CLAUDE.md — Macro Analysis Project Reference

## What this project does

Multi-agent macroeconomic researcher built on LangGraph. It ingests data from multiple
sources — FRED CSV files, research PDFs, the FRED REST API, and arbitrary web pages —
into a pgvector-backed Postgres database. A five-tool ReAct retrieval agent answers
questions from analysis agents using semantic search, with automatic fallback to live
data sources when the DB doesn't have an answer.

---

## Architecture

```
langgraph.json
  └── src/graph/graph.py              # LangGraph entrypoint — runs startup ingest,
        │                             # starts file watcher, compiles main_graph
        └── src/RAG/retreival_agent.py  # Five-tool ReAct researcher agent
              ├── search_pgvector          (semantic search, always first)
              ├── get_latest_data          (most-recent N rows, sorted by date)
              ├── fetch_fred_api           (live FRED REST API pull → index)
              ├── fetch_web_page           (Firecrawl / httpx scrape → index)
              └── search_web_news          (Firecrawl topic search → index)

src/config.py             # LLM, embeddings, PGVector stores, SQLRecordManager,
                          # FRED_API_KEY, FIRECRAWL_API_KEY, WATCH_DIRS
src/RAG/ingest_state.py   # SHA-256 hash sidecar for incremental ingest tracking
src/RAG/watcher.py        # watchdog daemon — auto-indexes file changes at runtime
src/RAG/fred_data_ingest.py  # CLI: run_incremental_ingestion / run_full_ingestion
src/RAG/memory.py         # SQLAlchemy ORM for analysis_records table (vector search)
src/RAG/sources/
  ├── base.py             # BaseSource ABC + IndexingResult dataclass
  ├── local_file_source.py  # CSV / PDF / TXT ingestion (used by watcher)
  ├── fred_api_source.py    # FRED REST API live pull
  └── web_source.py         # Firecrawl + httpx web scraper
```

**Data flow**:
1. At startup `graph.py` runs `run_incremental_ingestion()` (catches offline additions)
2. `graph.py` starts the watchdog daemon thread — new/changed files auto-index in ~2s
3. Run `langgraph dev` (or `make run`) to start the LangGraph API server
4. Analysis agents send messages to the `my_agent` graph endpoint
5. The researcher agent queries PGVector first; falls back to live sources as needed

---

## Environment Variables (`.env`)

| Variable | Purpose |
|---|---|
| `CLOD_API_KEY` | API key for the LLM (Clod.io OpenAI-compatible endpoint) — **required** |
| `SIMPLE_AGENT_MODEL` | Model string passed to `ChatOpenAI`, e.g. `anthropic:claude-sonnet-4-6` |
| `GEMINI_API_KEY` | Google Gemini API key for `GoogleGenerativeAIEmbeddings` — **required** |
| `DB_CONNECTION_STRING` | Full postgres connection string, e.g. `postgresql+psycopg://user:pass@localhost:5432/macro_db` |
| `LANGSMITH_API_KEY` | Optional — enables LangSmith tracing |
| `FRED_API_KEY` | Optional — enables live FRED REST API pulls. Free at fred.stlouisfed.org |
| `FIRECRAWL_API_KEY` | Optional — enables Firecrawl web scraping. Falls back to httpx if unset |
| `WATCH_DIRS` | Comma-separated dirs to watch for new files (default: `fred_fed_data,research_docs`) |

---

## Common Commands

```bash
make dev              # uv sync (install all deps including dev)
make run              # uv run langgraph dev  (starts LangGraph API server)
make test             # pytest tests/unit_tests
make lint             # ruff check
make format           # ruff format

# Incremental ingestion (only new/changed files — fast, safe to run anytime)
uv run python src/RAG/fred_data_ingest.py

# Full re-index (all files unconditionally — use for resets)
uv run python src/RAG/fred_data_ingest.py full

# Smoke test the graph directly
uv run python src/graph/graph.py
```

---

## Database

- **Image**: `pgvector/pgvector:pg17`
- **Extension**: `vector` (enabled automatically by `langchain-postgres`)
- **Tables managed by langchain-postgres**: `langchain_pg_collection`, `langchain_pg_embedding`
- **Tables managed by langchain-classic**: `upsertion_record` (dedup/indexing log)
- **Tables managed by memory.py**: `analysis_records` (stores `Analysis_Profile` results with embeddings)

### Connection string format
```
postgresql+psycopg://macro_user:macro_pass@localhost:5432/macro_db
```
Use `postgresql+psycopg` (sync) — `langchain-postgres` handles async internally via `async_mode=True`.

---

## Embeddings

- **Model**: `gemini-embedding-2` via `GoogleGenerativeAIEmbeddings`
- **Dimension**: **1536** (`output_dimensionality=1536` in `config.py`)
- `EMBEDDING_DIM` in `memory.py` must match this value
- Collection name: `my_docs_v5` (change `COLLECTION_NAME` in `config.py` to force a fresh collection)

---

## Known Gotchas

1. **`langchain_ollama` is not installed** — it's commented out in `config.py`. Do not un-comment
   the import at the top of the file without first running `uv add langchain-ollama`.

2. **`src/` must be on `sys.path`** — the LangGraph server runs from the project root, so
   `src/graph/graph.py` patches `sys.path` at import time. Do not remove this.

3. **`langgraph.json` entrypoint** is `./src/graph/graph.py:main_graph`, not `./src/graph.py`.

4. **`get_latest_data` is async** — it uses `await vector_store.asimilarity_search(...)`.
   The `vector_store` in `config.py` must have `async_mode=True`. The sync store
   (`vector_store_sync`) is used for the standard retriever tool.

5. **Batch size during ingestion** — `fred_data_ingest.py` sleeps 3 seconds between 500-row
   batches to avoid Gemini API rate limits. Adjust `time.sleep` if needed.

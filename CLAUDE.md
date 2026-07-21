# CLAUDE.md — Macro Analysis Project Reference

## What this project does

Multi-agent macroeconomic analysis system built on LangGraph. It ingests FRED CSV data into a
pgvector-backed Postgres database and exposes a retrieval agent that answers questions about
macroeconomic indicators (CPI, Fed Funds Rate, Treasury yields, etc.) using semantic search.

---

## Architecture

```
langgraph.json
  └── src/graph/graph.py            # LangGraph entrypoint — compiles main_graph
        └── src/retreival_agent.py  # ReAct agent with two tools:
              ├── search_fred_macro_data  (sync PGVector retriever, top-20)
              └── get_latest_data         (async PGVector search, sorted by date)

src/config.py             # LLM, embeddings, PGVector stores, SQLRecordManager
src/memory.py             # SQLAlchemy ORM for analysis_records table (vector search)
src/fred_data_ingest.py   # One-shot script: reads CSV files → indexes into pgvector
```

**Data flow**:
1. Run `fred_data_ingest.py` once to populate the vector store from `fred_fed_data/*.csv`
2. Run `langgraph dev` (or `make run`) to start the LangGraph API server
3. Send messages to the `my_agent` graph endpoint

---

## Environment Variables (`.env`)

| Variable | Purpose |
|---|---|
| `CLOD_API_KEY` | API key for the LLM (Clod.io OpenAI-compatible endpoint) — **required** |
| `SIMPLE_AGENT_MODEL` | Model string passed to `ChatOpenAI`, e.g. `anthropic:claude-sonnet-4-6` |
| `GEMINI_API_KEY` | Google Gemini API key for `GoogleGenerativeAIEmbeddings` — **required** |
| `DB_CONNECTION_STRING` | Full postgres connection string, e.g. `postgresql+psycopg://user:pass@localhost:5432/macro_db` |
| `LANGSMITH_API_KEY` | Optional — enables LangSmith tracing |

---

## Common Commands

```bash
make dev              # uv sync (install all deps including dev)
make run              # uv run langgraph dev  (starts LangGraph API server)
make test             # pytest tests/unit_tests
make lint             # ruff check
make format           # ruff format

# One-time data ingestion
uv run python src/fred_data_ingest.py

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

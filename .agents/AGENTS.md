# AGENTS.md — Macro Analysis Researcher: Project-Scoped Agent Rules

## System Purpose

This repository is a **living macroeconomic researcher**, not a one-shot ingest script.
Its job is to hold research information and answer questions from analysis agents with
the most accurate, most recent data available from three tiers of sources:

1. **PGVector database** — the canonical, indexed store of all research content
2. **Local files** — CSV data files, research PDFs, and text notes on disk
3. **Live sources** — FRED REST API and Firecrawl / httpx web fetching

---

## RAG Tool Routing Rules

When working within or extending the retrieval agent, always follow this priority order:

| Priority | Tool | When to use |
|---|---|---|
| 1st | `search_pgvector` | Always try first — fast, cached, covers all indexed content |
| 2nd | `get_latest_data` | "Most recent" / "current" queries for a known metric |
| 3rd | `fetch_fred_api` | Series not yet in DB, or user explicitly wants live FRED data |
| 4th | `fetch_web_page` | Specific URL provided (articles, Fed statements, reports) — uses Firecrawl / httpx |
| 5th | `search_web_news` | Broad macro topic, no specific URL, needs current news — uses Tavily |

**Never skip `search_pgvector`.** Every live fetch indexes its result into PGVector, so
the first call is expensive and subsequent ones are free.

---

## Source of Truth Contract

PGVector is the canonical store. Every document fetched from a live source **must** be
indexed before being returned to the caller. This is enforced in `BaseSource.fetch()`.

Do not bypass `index()` when adding new sources. Using `vector_store.add_documents()`
directly will skip the `SQLRecordManager` deduplication logic and cause duplicate embeddings.

---

## File Watcher Contract

Any file dropped into `fred_fed_data/` or `research_docs/` (configurable via `WATCH_DIRS`)
will be auto-indexed within ~2 seconds while the LangGraph server is running.

- **Do not** call `run_full_ingestion()` or `run_incremental_ingestion()` manually
  inside agent code — the watcher handles this automatically.
- The watcher starts as a daemon thread at `graph.py` import time (i.e., at server startup).
- An incremental scan also runs at startup to catch files added while the server was offline.

---

## Metric Naming Conventions

- FRED series IDs are **always uppercase**: `FEDFUNDS`, `DFF`, `CPILFESL`, `DGS2`, `UNRATE`, `GDP`.
- Web-sourced documents use the **URL as the `source` metadata key**.
- File-sourced documents use the **absolute file path as the `source` metadata key**.
- API-sourced documents use `fred_api:{SERIES_ID}:{date}` as the `source` key.

This ensures `SQLRecordManager` deduplication works correctly across all source types.

---

## Embedding Dimension Invariant

The embedding dimension is **always 1536** (`EMBEDDING_DIM` in `config.py`).

**Never change `EMBEDDING_DIM`** without:
1. Dropping and recreating the `langchain_pg_collection` / `langchain_pg_embedding` tables.
2. Deleting `ingest_state.json` to force a full re-index.
3. Updating the `query_embedding` column type in `analysis_records` (`memory.py`).

---

## Adding a New Data Source

Follow this checklist when creating a new `BaseSource` subclass:

1. Create `src/RAG/sources/your_source.py` inheriting from `BaseSource`.
2. Implement `fetch(query, **kwargs) -> list[Document]` — must call `index()` before returning.
3. Implement `ingest_all() -> IndexingResult` — bulk re-index for startup.
4. Add a unique `source` metadata key to every `Document` (see naming conventions above).
5. Export the class from `src/RAG/sources/__init__.py`.
6. Instantiate the source as a module-level singleton in `retreival_agent.py`.
7. Wrap `fetch()` with a `@tool`-decorated function and add it to the agent's tool list.
8. Update the agent system prompt to explain when to use the new tool.
9. Document the source in `src/RAG/rag_orchestration.md`.
10. Write a unit test in `tests/unit_tests/test_your_source.py`.

---

## Standard Document Metadata Schema

Every `Document` indexed into PGVector must include at minimum:

```python
{
    "source": str,        # Unique key used for deduplication (file path, URL, or API key)
    "file_type": str,     # One of: "csv", "pdf", "txt", "fred_api", "web"
}
```

Time-series documents (CSV, FRED API) should also include:

```python
{
    "metric": str,        # FRED series ID or indicator name
    "date": str,          # ISO 8601 date string: "YYYY-MM-DD"
    "year": int,          # Parsed year for range filtering
}
```

Web documents should also include:

```python
{
    "url": str,           # Full URL
    "domain": str,        # Parsed domain (e.g. "www.federalreserve.gov")
    "fetched_at": str,    # ISO 8601 UTC timestamp
}
```

---

## Known Gotchas (Inherited)

1. **`langchain_ollama` is not installed** — it's commented out in `config.py`. Do not
   un-comment the import without first running `uv add langchain-ollama`.

2. **`src/` must be on `sys.path`** — `src/graph/graph.py` patches `sys.path` at import
   time. Do not remove this.

3. **`langgraph.json` entrypoint** is `./src/graph/graph.py:main_graph`.

4. **`get_latest_data` is async** — uses `await vector_store.asimilarity_search(...)`.
   The async `vector_store` in `config.py` must have `async_mode=True`.

5. **Gemini embedding rate limit** — batch size 500 with 3-second sleep between batches.
   Adjust `time.sleep` in `fred_data_ingest.py` if you have a higher quota tier.

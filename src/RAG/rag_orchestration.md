# RAG Orchestration Reference

Technical reference for the RAG researcher layer in `src/RAG/`.
See `.agents/AGENTS.md` for behavioural rules and routing contracts.

---

## Module Map

```
src/RAG/
├── retreival_agent.py       # ReAct orchestrator — assembles tools + agent
├── fred_data_ingest.py      # CLI + importable ingest entry points
├── ingest_state.py          # SHA-256 hash sidecar for incremental ingest
├── watcher.py               # watchdog daemon — auto-indexes file changes
├── memory.py                # Long-term analysis_records table (SQLAlchemy + pgvector)
└── sources/
    ├── __init__.py           # Re-exports all source classes
    ├── base.py               # BaseSource ABC + IndexingResult dataclass
    ├── local_file_source.py  # CSV / PDF / TXT ingestion
    ├── fred_api_source.py    # FRED REST API live pull
    └── web_source.py         # Firecrawl + httpx web scraper
```

---

## Tool Routing Table

| Tool | Module | Trigger Condition | Backend | Return Shape |
|---|---|---|---|---|
| `search_pgvector` | `retreival_agent.py` | Default — all queries | PGVector | `list[Document]` via LangChain retriever |
| `get_latest_data` | `retreival_agent.py` | "Latest" / "most recent" metric value | PGVector | `str` — N rows sorted by date |
| `fetch_fred_api` | `sources/fred_api_source.py` | Live FRED series not in DB or explicitly requested | FRED REST API | `str` — N most-recent observations |
| `fetch_web_page` | `sources/web_source.py` | Specific URL provided | Firecrawl → httpx fallback | `str` — extracted page text |
| `search_web_news` | `retreival_agent.py` | Broad macro topic, no URL | **Tavily** | `str` — synthesised answer + result snippets |

---

## Ingest Lifecycle

```
Server starts (graph.py imported)
        │
        ▼
run_incremental_ingestion()          ← catches files added while server was offline
        │
        ├── for each watched dir:
        │       get_new_or_changed_files()   ← compares SHA-256 against ingest_state.json
        │       load_file(path)              ← CSV / PDF / TXT loader
        │       index(docs, cleanup="incremental")   ← PGVector + SQLRecordManager
        │       mark_file_ingested(path)     ← updates ingest_state.json
        │
        ▼
start_watching(dirs, local_source)   ← daemon thread starts
        │
        └── watchdog Observer monitors dirs for created/modified/moved events
                │
                ▼ (event fires)
            _ResearchFileHandler._handle()
                │   debounce check (1.5 s window)
                ▼
            local_source.ingest_file(path)
                │
                ▼
            PGVector updated ← agent queries now return fresh data


Agent query arrives
        │
        ▼
search_pgvector  ──► hit? ──► return docs
        │
        no hit
        ▼
fetch_fred_api / fetch_web_page / search_web_news
        │
        ▼
results indexed into PGVector
        │
        ▼
return to agent (and cached for next query)
```

---

## Source Registry

To register a new source so the orchestrator uses it:

1. Subclass `BaseSource` in `src/RAG/sources/your_source.py`.
2. Export it from `src/RAG/sources/__init__.py`.
3. Create a module-level singleton in `retreival_agent.py`.
4. Wrap `.fetch()` in a `@tool` function and add it to the `create_agent(tools=[...])` call.

The agent discovers tools at import time — no dynamic registration needed.

---

## Metadata Schema

### All Documents

| Key | Type | Required | Description |
|---|---|---|---|
| `source` | `str` | ✅ | Unique deduplication key (file path, URL, or `fred_api:{SERIES}:{date}`) |
| `file_type` | `str` | ✅ | `"csv"`, `"pdf"`, `"txt"`, `"fred_api"`, or `"web"` |

### Time-Series Documents (CSV / FRED API)

| Key | Type | Description |
|---|---|---|
| `metric` | `str` | FRED series ID or indicator name (uppercase) |
| `date` | `str` | ISO 8601: `"YYYY-MM-DD"` |
| `year` | `int` | Parsed year for range filtering |
| `value` | `str` | Raw observation value |

### Web Documents

| Key | Type | Description |
|---|---|---|
| `url` | `str` | Full URL |
| `domain` | `str` | Parsed domain (e.g. `"www.federalreserve.gov"`) |
| `fetched_at` | `str` | ISO 8601 UTC timestamp |

---

## Deduplication Strategy

`SQLRecordManager` + `index(cleanup="incremental", key_encoder="sha256")` provides
content-hash-based deduplication:

- **Source key** (`source_id_key="source"`) identifies the origin of a document.
- **SHA-256 hash** of the document content determines whether it has changed.
- If `source` key + content hash already exist in `upsertion_record`, the document
  is skipped (`num_skipped` increments).
- `cleanup="incremental"` removes stale embeddings for documents whose source key
  exists but whose content hash has changed (i.e., the file was updated).

`ingest_state.py` provides a second deduplication layer at the *file* level,
preventing re-loading a file's contents into memory if its SHA-256 hasn't changed.
This makes startup scans very fast even with hundreds of files.

---

## Known Rate Limits

| Source | Limit | Mitigation |
|---|---|---|
| Gemini Embedding API | ~500 requests/min (free tier) | 3-second sleep between 500-doc batches in `fred_data_ingest.py` |
| FRED REST API (no key) | 120 requests/day | Always set `FRED_API_KEY` for production use |
| FRED REST API (with key) | 120 requests/min | Sufficient for on-demand + daily refresh |
| Firecrawl | Plan-dependent (credits) | Results cached in PGVector; repeated queries don't re-fetch |
| httpx fallback | No API limit | Subject to target server rate-limiting and robots.txt |

---

## Failure Modes and Fallbacks

| Failure | Behaviour |
|---|---|
| `FRED_API_KEY` not set | `fetch_fred_api` returns an instructive error message; agent falls back to `search_pgvector` |
| `FIRECRAWL_API_KEY` not set | `search_web_news` returns error; `fetch_web_page` falls back to httpx + BeautifulSoup |
| Firecrawl SDK throws | `web_source._scrape()` catches and retries with httpx fallback |
| FRED API HTTP error | `_pull_series()` logs error and returns `[]`; agent sees "No data returned" message |
| File watcher crash | Logged as warning; server continues — startup incremental ingest catches changes on restart |
| Startup ingest failure | Logged as warning; server still starts; existing DB data remains available |
| PGVector connection lost | Propagates as exception — handled by LangGraph retry/error logic |

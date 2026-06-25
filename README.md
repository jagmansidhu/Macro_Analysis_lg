# Macro Analysis

A multi-agent system for macroeconomic analysis, built on LangGraph. Agents retrieve data from multiple sources, analyze it, and backtest the analysis before producing a final output.

## Why this exists

Macro calls require pulling data from scattered sources (FRED time series, Fed reports, market news), reading through it, forming a thesis, and then checking whether the thesis holds against historical precedent. Each of those steps is a distinct job. This project splits them across specialized agents that hand work to each other through a LangGraph orchestrator, so the full pipeline runs as a single query.

## Example: "Where is inflation heading over the next 6 months?"

You send that question to the LangGraph endpoint. The orchestrator takes it from there.

**Step 1 — Retrieval.** The orchestrator passes the question to the retrieval orchestrator, which fans out to three agents:
- The FRED agent pulls the last 12 months of Core CPI (CPILFESL), fed funds rate (DFF), and 2-year Treasury yields (DGS2).
- The news agent pulls recent articles on inflation expectations, wage growth, and consumer spending.
- The Fed reports agent pulls the latest FOMC minutes and any relevant Beige Book sections on pricing pressures.

All three results merge into a single data package.

**Step 2 — Analysis.** The orchestrator sends the merged data to the analyzer agent. The analyzer reads through it and returns a structured thesis:
```json
{
  "direction": "declining",
  "metric": "CPILFESL",
  "timeframe": "6 months",
  "confidence": 0.7,
  "reasoning": "Core CPI has decelerated for 3 consecutive months. FOMC minutes signal no further hikes. 2-year yields have dropped 40bps since March, pricing in cuts."
}
```

**Step 3 — Backtest.** The orchestrator sends the thesis to the test agent. The test agent queries for historical periods where Core CPI showed a similar 3-month deceleration with fed funds at a comparable level. It finds 4 matching periods: in 3 of them, CPI continued declining over the following 6 months. In 1, it reversed due to an oil shock. The test agent returns a pass with caveats.

**Step 4 — Feedback (if needed).** If the test agent had rejected the thesis (say, 3 of 4 historical matches contradicted it), the orchestrator would route the failure evidence back to the analyzer. The analyzer would revise its thesis given the new context and the loop would repeat. This continues until the thesis passes or hits a retry limit.

**Step 5 — Output.** The orchestrator assembles the final response: the thesis, the supporting data, the backtest results, and any caveats. That response comes back to you through the LangGraph dev UI or API.

## Agent architecture

```
                         ┌──────────────────────┐
                         │   Orchestrator Agent  │
                         │                       │
                         │  Owns the full        │
                         │  pipeline. Routes     │
                         │  data between agents  │
                         │  and decides when to  │
                         │  loop back.           │
                         └───┬──────┬────────┬───┘
                             │      │        │
                ┌────────────┘      │        └────────────┐
                ▼                   ▼                      ▼
   ┌─────────────────────┐ ┌─────────────────┐  ┌──────────────────┐
   │ Retrieval            │ │ Analyzer Agent  │  │   Test Agent     │
   │ Orchestrator         │ │                 │  │                  │
   │                      │ │ Takes retrieved │  │  Backtests the   │
   │ Dispatches to        │ │ data, forms a   │  │  analyzer's      │
   │ source-specific      │ │ macro thesis    │  │  thesis against  │
   │ retrieval agents     │ │ with structured │  │  historical data │
   │ and merges results.  │ │ output.         │  │  to check if the │
   │                      │ │                 │  │  reasoning holds.│
   │  ┌───────────────┐   │ └─────────────────┘  └──────────────────┘
   │  │ FRED Agent    │   │
   │  │ (working)     │   │
   │  ├───────────────┤   │
   │  │ News Agent    │   │
   │  │ (planned)     │   │
   │  ├───────────────┤   │
   │  │ Fed Reports   │   │
   │  │ Agent         │   │
   │  │ (planned)     │   │
   │  └───────────────┘   │
   └──────────────────────┘
```

## Current state

### FRED retrieval agent (working)

The FRED retrieval agent queries time-series data stored in pgvector. It has two tools:

- `search_fred_macro_data` does semantic similarity search (k=20) for historical and trend questions.
- `get_latest_data` filters by metric name and sorts by date, so "most recent CPI" returns the actual latest row instead of whatever embedding lands closest.

Data gets into pgvector through `fred_data_ingest.py`, which reads every CSV in `fred_fed_data/`, creates one LangChain Document per row, and indexes them in 500-row batches. A `SQLRecordManager` with SHA-256 hashing handles deduplication on re-runs.

Four FRED series are loaded today:

| File               | Series   | What it is                                |
| ------------------ | -------- | ----------------------------------------- |
| `CPILFESL (1).csv` | CPILFESL | Core CPI (all items less food and energy) |
| `DFF.csv`          | DFF      | Daily effective federal funds rate        |
| `DGS2.csv`         | DGS2     | 2-year Treasury constant maturity yield   |
| `FEDFUNDS.csv`     | FEDFUNDS | Monthly effective federal funds rate      |

The LangGraph graph is a single node (`retrieval_worker`) wired START to END. LangSmith tracing works when `LANGSMITH_API_KEY` is set.

### Not yet built

- **Retrieval orchestrator.** Sits between the main orchestrator and the source-specific retrieval agents. Decides which sources to query for a given question, dispatches to the right agents, and merges the results.
- **News retrieval agent.** Pulls market news from a feed TBD. Needs recency weighting so recent articles rank higher than old ones with similar text.
- **Fed reports retrieval agent.** Ingests FOMC minutes, Beige Book summaries, and Fed speeches. Chunking strategy will differ from FRED (section-level instead of row-level).
- **Analyzer agent.** Receives the merged retrieval data and produces a structured thesis: direction, metric, timeframe, confidence. Structured output (not freeform text) so the test agent can run a concrete backtest without interpreting prose.
- **Test agent.** Takes the analyzer's structured thesis, queries the retrieval agents for historical periods with similar conditions, and checks whether the predicted outcome occurred.
- **Orchestrator agent.** Runs the full pipeline and owns the feedback loop. If the test agent rejects a thesis, the orchestrator routes the failure evidence back to the analyzer for revision instead of stopping.

## TODO

- [ ] Build the retrieval orchestrator to dispatch across source-specific agents
- [ ] Add market news retrieval agent with recency-weighted search
- [ ] Add Fed reports retrieval agent (FOMC minutes, Beige Book, speeches) with section-level chunking
- [ ] Define structured output schema for the analyzer (direction, metric, timeframe, confidence)
- [ ] Build analyzer agent with structured output
- [ ] Build test agent that backtests structured theses against historical retrieval data
- [ ] Build orchestrator agent with feedback loop (test failure → re-analyze)
- [ ] Add `interrupt_before` checkpoint between analyzer and test agent for human review during development
- [ ] Store past analyses (thesis + backtest result + date) so the system can track its own accuracy across runs

## Stack

| Layer           | Tool                                                               |
| --------------- | ------------------------------------------------------------------ |
| LLM             | Clod proxy (OpenAI-compatible), model set via `SIMPLE_AGENT_MODEL` |
| Embeddings      | Google `gemini-embedding-2`, 1536 dimensions                       |
| Vector store    | PostgreSQL + pgvector via `langchain-postgres`                     |
| Agent framework | LangGraph + LangChain                                              |
| Tracing         | LangSmith (optional)                                               |
| Package manager | uv                                                                 |

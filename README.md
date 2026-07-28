# Macro Analysis: Autonomous Economic Researcher

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-LangGraph-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Vector Store](https://img.shields.io/badge/Vector_Store-PGVector-blue.svg)](https://github.com/pgvector/pgvector)

A multi-agent system for macroeconomic analysis built on LangGraph. This system autonomously retrieves financial data from distributed sources (FRED time series, Fed reports, market news), synthesises a thesis, and backtests its analysis against historical precedent to produce high-confidence macro forecasts.

## Overview

Traditional macro analysis requires manually pulling scattered data, forming a thesis, and verifying it against historical regimes. This project automates that workflow by splitting responsibilities across specialised agents that pass data and context through a LangGraph orchestrator.

### Core Capabilities
- **Multi-Source Ingestion**: Automatically ingests and indexes FRED CSV files, research PDFs, and arbitrary web pages into a `pgvector` store.
- **Dynamic Retrieval**: A 5-tool ReAct agent that queries local vectors, pulls live data from the FRED REST API, and scrapes web news using Firecrawl/Tavily.
- **Automated Backtesting (Planned)**: Validates generated macro theses against historical time-series data to ensure reasoning holds.

## Architecture

```text
                         ┌──────────────────────┐
                         │  Orchestrator Agent  │
                         │                      │
                         │ Owns the pipeline,   │
                         │ routes data, manages │
                         │ feedback loops.      │
                         └───┬──────┬───────┬───┘
                             │      │       │
                ┌────────────┘      │       └─────────────┐
                ▼                   ▼                     ▼
   ┌─────────────────────┐ ┌─────────────────┐ ┌──────────────────┐
   │   Retrieval Agent   │ │ Analyzer Agent  │ │    Test Agent    │
   │                     │ │                 │ │                  │
   │ 5-tool ReAct agent  │ │ Forms a macro   │ │ Backtests thesis │
   │ fetching from DB,   │ │ thesis with     │ │ against history  │
   │ FRED API, and web   │ │ structured      │ │ to ensure it     │
   │ (Firecrawl/Tavily)  │ │ output.         │ │ holds up.        │
   └─────────────────────┘ └─────────────────┘ └──────────────────┘
```

## Getting Started

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (for dependency management)
- PostgreSQL 17+ with the `pgvector` extension

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/macro-analysis.git
   cd macro-analysis
   ```

2. **Configure environment:**
   Create a `.env` file in the project root:
   ```env
   CLOD_API_KEY=your_llm_api_key
   GEMINI_API_KEY=your_gemini_api_key
   DB_CONNECTION_STRING=postgresql+psycopg://user:pass@localhost:5432/macro_db
   # Optional
   FRED_API_KEY=your_fred_key
   FIRECRAWL_API_KEY=your_firecrawl_key
   TAVILY_API_KEY=your_tavily_key
   ```

3. **Install dependencies:**
   ```bash
   make dev
   ```

4. **Run the server:**
   ```bash
   make run
   ```

### Common Commands

```bash
make test             # Run unit tests via pytest
make lint             # Check formatting via ruff
make format           # Auto-format via ruff

# Trigger an incremental data ingestion manually
uv run python src/RAG/fred_data_ingest.py
```

## Project Structure

- `src/graph/`: Contains the LangGraph definition and entrypoint (`graph.py`).
- `src/RAG/`: The core retrieval system.
  - `retreival_agent.py`: The ReAct orchestrator and tool definitions.
  - `sources/`: Source-specific connectors (`local_file_source`, `fred_api_source`, `web_source`).
  - `memory.py` / `ingest_state.py` / `watcher.py`: Ingestion, deduplication, and DB memory.
- `tests/`: Comprehensive unit and integration test suites.

## System Stack

| Component       | Technology                                                         |
|-----------------|--------------------------------------------------------------------|
| **LLM**         | Clod proxy (OpenAI-compatible) via `SIMPLE_AGENT_MODEL`            |
| **Embeddings**  | Google `gemini-embedding-2` (1536 dimensions)                      |
| **Database**    | PostgreSQL + pgvector (managed by `langchain-postgres`)            |
| **Orchestration**| LangGraph + LangChain                                             |
| **Tooling**     | uv (package manager), ruff (linting/formatting), pytest (testing)  |

## Roadmap

- [ ] **Analyzer Agent**: Define the structured output schema (direction, metric, timeframe, confidence) and implement the thesis generation.
- [ ] **Test Agent**: Build the logic to query historical regimes that match current conditions to validate the analyzer's thesis.
- [ ] **Feedback Loop**: Implement the `Orchestrator Agent` to route test failures back to the analyzer for revision.
- [ ] **Human-in-the-Loop**: Add an `interrupt_before` LangGraph checkpoint between the analyzer and test agent for manual review during development.
- [ ] **Long-Term Memory**: Store past analyses (thesis + backtest result + date) so the system can evaluate its own predictive accuracy over time.

# Project Intent: Macro Analysis

## 1. The "Why": Purpose of the System
Macroeconomic analysis relies heavily on historical precedent, complex data synthesis, and objective backtesting. Current LLMs, when asked macro questions, often hallucinate, rely on outdated training weights, or provide generic prose without verifying if their reasoning has ever held true in real life. 

This project is built to solve that. The goal is to construct a **living, autonomous macroeconomic researcher** using a multi-agent LangGraph architecture. Rather than just returning retrieved data (RAG), the system must explicitly form a hypothesis (a thesis) and automatically **backtest that thesis against historical data** to prove or disprove its own logic before returning an answer to the user.

## 2. The "How It Will Look": Target End State
When fully built, the LangGraph orchestrator will run a continuous loop of retrieval, analysis, and verification:

1. **User Query**: e.g., *"Where is inflation heading over the next 6 months?"*
2. **Retrieval**: The `Retrieval Agent` (currently built) fetches live FRED data, local DB vectors, and web news to build a context package.
3. **Analysis**: The `Analyzer Agent` reads the package and outputs a strictly typed structured thesis (e.g., `{"direction": "declining", "metric": "CPI", "timeframe": "6 months"}`).
4. **Backtesting**: The `Test Agent` intercepts the thesis, queries the DB for similar historical conditions (e.g., "when CPI dropped 3 months in a row while Fed Funds were flat"), and checks if the expected outcome actually happened.
5. **Feedback Loop**: If the thesis fails the backtest, the orchestrator sends the failure evidence back to the Analyzer for a rewrite.
6. **Final Output**: The user receives a highly confident, backtested thesis with data citations.

## 3. Roadmap and Future Plans (Steps)

The following steps define the critical path to reaching the target end state. **Agents working on this repository should refer to this list to stay aligned with the project's ultimate goals.**

### Phase 1: Retrieval (Completed)
- [x] PGVector database setup with 1536-dim Gemini embeddings.
- [x] File watcher daemon for auto-ingesting CSV/PDF/TXT files.
- [x] `Retrieval Agent` (5-tool ReAct) pulling from DB, FRED REST API, Firecrawl, and Tavily.

### Phase 2: Analysis Engine (Next Up)
- [ ] **Define Structured Output**: Create Pydantic schemas for the Analyzer Agent's thesis (requires explicit fields for `direction`, `metric`, `timeframe`, and `confidence`). Freeform text cannot be reliably backtested.
- [ ] **Build Analyzer Agent**: Create the LangGraph node that takes the `Retrieval Agent`'s output and strictly conforms to the structured thesis schema.

### Phase 3: Verification Engine
- [ ] **Build Test Agent**: This agent will take the structured thesis and formulate historical queries. It will use the `Retrieval Agent`'s tools to look up past data regimes that match the thesis conditions.
- [ ] **Implement Backtesting Logic**: The Test Agent must output a binary Pass/Fail along with a markdown summary of historical evidence.

### Phase 4: Orchestration & Feedback
- [ ] **Build Orchestrator Loop**: Wire LangGraph so a "Fail" from the Test Agent routes back to the Analyzer Agent with the historical contradictions injected into its prompt.
- [ ] **Human-in-the-Loop**: Add an `interrupt_before` node before the final output so a human can approve the analysis during development.
- [ ] **Long-Term Memory**: Write final approved analyses to a persistent memory table so the system tracks its own accuracy and doesn't repeat past analytical mistakes.

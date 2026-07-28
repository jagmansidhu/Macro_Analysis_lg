"""LangGraph entrypoint — compiles the main_graph.

On import this module:
  1. Runs an incremental ingest to catch any files added while the server was offline.
  2. Starts the watchdog daemon thread to monitor configured directories.
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.graph import StateGraph, MessagesState, START, END

from RAG.retreival_agent import retrieval_agent
from RAG.fred_data_ingest import run_incremental_ingestion
from RAG.watcher import start_watching
from RAG.sources.local_file_source import LocalFileSource
from config import vector_store_sync, record_manager, WATCH_DIRS

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
try:
    logger.info("Running incremental ingestion on startup...")
    run_incremental_ingestion()
except Exception as exc:
    # Non-fatal — the server should still start even if ingest fails
    logger.warning("Startup incremental ingestion failed: %s", exc)

try:
    abs_dirs = [PROJECT_ROOT / d for d in WATCH_DIRS]
    local_source = LocalFileSource(
        directories=abs_dirs,
        vector_store_sync=vector_store_sync,
        record_manager=record_manager,
    )
    start_watching(directories=abs_dirs, local_source=local_source)
except Exception as exc:
    logger.warning("File watcher could not start: %s", exc)

builder = StateGraph(MessagesState)

builder.add_node("retrieval_worker", retrieval_agent)

builder.add_edge(START, "retrieval_worker")
builder.add_edge("retrieval_worker", END)

main_graph = builder.compile()


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    print("Testing the macro-economic RAG researcher pipeline...\n")

    inputs = {"messages": [HumanMessage(content="What was the DFF in 2020?")]}

    for event in main_graph.stream(inputs, stream_mode="values"):
        event["messages"][-1].pretty_print()
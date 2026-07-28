"""Data ingestion entry points for the RAG pipeline.

Two modes:
  run_full_ingestion()        — re-index every file unconditionally (original behaviour)
  run_incremental_ingestion() — only index new or changed files (used at startup)

Both are importable so graph.py and watcher.py can call them directly.
"""

import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from langchain_core.documents import Document
from langchain_core.indexing import index
from langsmith import traceable

from config import vector_store_sync, record_manager, WATCH_DIRS
from RAG.sources.local_file_source import LocalFileSource, load_file, _load_csv as _csv_loader
from RAG.ingest_state import mark_file_ingested

# Backwards-compatibility re-export — test_ingest_parsing.py imports this directly
from RAG.sources.local_file_source import (  # noqa: F401
    _load_csv,
    _load_txt,
)


def load_local_directory(directory_path: str, glob_pattern: str):
    """Backwards-compatible shim — delegates to LocalFileSource loaders."""
    from RAG.sources.local_file_source import load_file
    from pathlib import Path

    docs = []
    for file_path in sorted(Path(directory_path).glob(glob_pattern)):
        if file_path.is_file():
            docs.extend(load_file(file_path))
    return docs


logger = logging.getLogger(__name__)


def _build_local_source() -> LocalFileSource:
    """Construct a LocalFileSource pointed at all configured watch directories."""
    abs_dirs = [PROJECT_ROOT / d for d in WATCH_DIRS]
    return LocalFileSource(
        directories=abs_dirs,
        vector_store_sync=vector_store_sync,
        record_manager=record_manager,
    )


@traceable(name="Incremental Data Ingestion")
def run_incremental_ingestion() -> None:
    """Index only new or changed files — fast, safe to run at every startup."""
    logger.info("Starting incremental ingestion...")
    source = _build_local_source()
    result = source.ingest_all()
    logger.info("Incremental ingestion complete: %s", result)
    print(f"Incremental ingestion complete: {result}")


@traceable(name="Full Data Ingestion")
def run_full_ingestion() -> None:
    """Re-index all files unconditionally (original behaviour, used for resets)."""
    logger.info("Starting full ingestion...")
    source = _build_local_source()

    # Gather all docs from all watched directories
    all_docs: list[Document] = []
    from RAG.sources.local_file_source import load_file

    for directory in source.directories:
        if not directory.exists():
            logger.warning("Directory does not exist, skipping: %s", directory)
            continue
        for file_path in sorted(directory.glob("*")):
            if file_path.suffix.lower() in {".csv", ".pdf", ".txt"}:
                docs = load_file(file_path)
                all_docs.extend(docs)
                logger.info("Loaded %d docs from %s", len(docs), file_path.name)

    print(f"Loaded {len(all_docs)} total documents.")

    batch_size = 500
    for i in range(0, len(all_docs), batch_size):
        batch = all_docs[i : i + batch_size]
        print(f"Processing batch {i // batch_size + 1} ({i} to {i + len(batch)})...")

        result = index(
            docs_source=batch,
            record_manager=record_manager,
            vector_store=vector_store_sync,
            cleanup=None,
            source_id_key="source",
            key_encoder="sha256",
            batch_size=batch_size,
        )
        print(f"Batch complete: {result}")
        time.sleep(3)  # Respect Gemini embedding API rate limits

    # Update the hash state so incremental ingestion knows these files are done
    for directory in source.directories:
        if not directory.exists():
            continue
        for file_path in sorted(directory.glob("*")):
            if file_path.suffix.lower() in {".csv", ".pdf", ".txt"}:
                mark_file_ingested(file_path)

    print("Full ingestion complete.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "incremental"
    if mode == "full":
        run_full_ingestion()
    else:
        run_incremental_ingestion()
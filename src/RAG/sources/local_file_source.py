"""LocalFileSource — watches and indexes local CSV, PDF, and TXT research files.

Supported formats:
  *.csv  — macroeconomic data (FRED-style: date column + value column)
  *.pdf  — research reports, whitepapers (text extracted via pypdf)
  *.txt  — plain text notes or data dumps
"""

import csv
import logging
import re
from pathlib import Path
from typing import Any

from langchain_classic.indexes import SQLRecordManager
from langchain_core.documents import Document
from langchain_core.indexing import index

from .base import BaseSource, IndexingResult
from RAG.ingest_state import (
    get_new_or_changed_files,
    mark_file_ingested,
)

logger = logging.getLogger(__name__)

DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Matches column names that end with an underscore + 4-digit year, e.g. "market_value_2003"
YEAR_COL_PATTERN = re.compile(r"^(.+)_(\d{4})$")

# Glob patterns matched per extension
_SUPPORTED_GLOBS = ["*.csv", "*.pdf", "*.txt"]


def _detect_date_column(fieldnames: list[str], sample_row: dict) -> str | None:
    for col in fieldnames:
        val = (sample_row.get(col) or "").strip()
        if DATE_PATTERN.match(val):
            return col
    return None


def _detect_year_columns(
    fieldnames: list[str],
) -> tuple[list[str], dict[int, list[tuple[str, str]]]]:
    """
    Identify wide-format year columns (e.g. ``market_value_2003``).

    Returns
    -------
    key_cols : list[str]
        Columns that are *not* year-suffixed (entity identifiers).
    year_map : dict[int, list[tuple[str, str]]]
        ``{year: [(column_name, metric_base_name), ...]}``, ordered by year.
    """
    key_cols: list[str] = []
    year_map: dict[int, list[tuple[str, str]]] = {}
    for col in fieldnames:
        m = YEAR_COL_PATTERN.match(col)
        if m:
            base, yr_str = m.group(1), m.group(2)
            yr = int(yr_str)
            year_map.setdefault(yr, []).append((col, base))
        else:
            key_cols.append(col)
    return key_cols, year_map


def _load_csv_wide(
    file_path: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> list[Document]:
    """
    Handle wide-format CSVs where years are encoded in column names.

    For each row × year combination, emit a single Document whose
    ``page_content`` summarises all metric values for that entity and year.
    Example output::

        In 2003, UNITED STATES (cntry_code=1007):
          market_value_residence = 0
          market_value_nationality = 181900
          mv_difference_nat_vs_res = 181900
    """
    docs: list[Document] = []
    key_cols, year_map = _detect_year_columns(fieldnames)

    if not year_map:
        logger.warning(
            "No date column and no year-suffixed columns in %s — skipping.",
            file_path.name,
        )
        return []

    logger.info(
        "Wide-format CSV detected: %s — %d years, %d key cols.",
        file_path.name,
        len(year_map),
        len(key_cols),
    )

    file_metric = re.sub(r"\s*\(\d+\)$", "", file_path.stem)

    for row in rows:
        # Build a human-readable entity label from all non-year key columns
        entity_parts = [f"{k}={row.get(k, '').strip()}" for k in key_cols]
        entity_label = ", ".join(p for p in entity_parts if p.split("=", 1)[1])

        for year in sorted(year_map):
            metric_lines: list[str] = []
            for col, base in year_map[year]:
                val = (row.get(col) or "").strip()
                if val:
                    metric_lines.append(f"  {base} = {val}")

            if not metric_lines:
                continue

            content = (
                f"In {year}, {entity_label}:\n" + "\n".join(metric_lines)
            )
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": str(file_path),
                        "metric": file_metric,
                        "date": f"{year}-01-01",
                        "year": year,
                        "file_type": "csv",
                        **{k: row.get(k, "").strip() for k in key_cols},
                    },
                )
            )
    return docs


def _load_csv(file_path: Path) -> list[Document]:
    docs: list[Document] = []
    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [n.strip() for n in (reader.fieldnames or []) if n]
            rows = list(reader)
            if not rows:
                return []

            date_col = _detect_date_column(reader.fieldnames, rows[0])
            if not date_col:
                # Fall back to wide-format (year-suffixed columns) handler
                return _load_csv_wide(file_path, rows, reader.fieldnames)

            val_col = next(
                (k for k in reader.fieldnames if k != date_col), None
            )
            if not val_col:
                return []

            metric_name = re.sub(r"\s*\(\d+\)$", "", file_path.stem)

            for row in rows:
                date = (row.get(date_col) or "").strip()
                value = (row.get(val_col) or "").strip()
                if not date or not value:
                    continue
                docs.append(
                    Document(
                        page_content=(
                            f"As of {date}, the value for macroeconomic "
                            f"indicator {metric_name} is {value}."
                        ),
                        metadata={
                            "source": str(file_path),
                            "metric": metric_name,
                            "date": date,
                            "year": int(date.split("-")[0]) if "-" in date else 0,
                            "file_type": "csv",
                        },
                    )
                )
    except Exception as exc:
        logger.error("Failed to read CSV %s: %s", file_path, exc)
    return docs


def _load_pdf(file_path: Path) -> list[Document]:
    """Extract text from a PDF using pypdf (one Document per page)."""
    docs: list[Document] = []
    try:
        from pypdf import PdfReader  # lazy import — optional dependency

        reader = PdfReader(str(file_path))
        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(file_path),
                        "file_type": "pdf",
                        "page": page_num,
                        "filename": file_path.name,
                    },
                )
            )
    except ImportError:
        logger.error("pypdf is not installed — cannot load PDF %s.", file_path.name)
    except Exception as exc:
        logger.error("Failed to read PDF %s: %s", file_path, exc)
    return docs


def _load_txt(file_path: Path) -> list[Document]:
    """Load a plain-text file as a single Document."""
    try:
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        return [
            Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "file_type": "txt",
                    "filename": file_path.name,
                },
            )
        ]
    except Exception as exc:
        logger.error("Failed to read TXT %s: %s", file_path, exc)
        return []


def load_file(file_path: Path) -> list[Document]:
    """Dispatch to the correct loader based on file extension."""
    ext = file_path.suffix.lower()
    if ext == ".csv":
        return _load_csv(file_path)
    if ext == ".pdf":
        return _load_pdf(file_path)
    if ext == ".txt":
        return _load_txt(file_path)
    logger.debug("Unsupported extension %s — skipping %s.", ext, file_path.name)
    return []


class LocalFileSource(BaseSource):
    """
    Indexes research files from one or more local directories.

    On ``ingest_all`` (called at startup) it performs an *incremental* scan:
    only files whose SHA-256 hash differs from the persisted state are processed.

    On ``fetch`` it immediately indexes a specific file, then returns its docs.
    This is the hook the file-watcher calls when a new/modified file is detected.
    """

    source_name = "local_file"

    def __init__(
        self,
        directories: list[str | Path],
        vector_store_sync,
        record_manager: SQLRecordManager,
        batch_size: int = 500,
    ):
        self.directories = [Path(d) for d in directories]
        self.vector_store_sync = vector_store_sync
        self.record_manager = record_manager
        self.batch_size = batch_size

    def fetch(self, query: str, **kwargs: Any) -> list[Document]:
        """
        Index a single file on demand (used by the file watcher).
        ``query`` is interpreted as the file path to ingest.
        """
        file_path = Path(query)
        if not file_path.exists():
            logger.warning("fetch() called with non-existent path: %s", query)
            return []
        docs = load_file(file_path)
        if docs:
            self._index_batch(docs, file_path)
        return docs

    def ingest_all(self) -> IndexingResult:
        """Incrementally index all new/changed files across all watched directories."""
        combined = IndexingResult()
        for directory in self.directories:
            if not directory.exists():
                logger.info("Watch dir does not exist yet, skipping: %s", directory)
                continue
            result = self._ingest_directory(directory)
            combined = combined.merge(result)
        logger.info("Incremental ingest complete: %s", combined)
        return combined

    def ingest_file(self, file_path: Path) -> IndexingResult:
        """Public helper — index a single file and update the state tracker."""
        docs = load_file(file_path)
        if not docs:
            return IndexingResult(num_skipped=1)
        result = self._index_batch(docs, file_path)
        mark_file_ingested(file_path)
        return result

    def _ingest_directory(self, directory: Path) -> IndexingResult:
        combined = IndexingResult()
        changed_files = get_new_or_changed_files(
            directory,
            glob_pattern="*",  # flat scan; watcher handles subdirs separately
        )
        # Filter to supported extensions
        changed_files = [
            p for p in changed_files
            if p.suffix.lower() in {".csv", ".pdf", ".txt"}
        ]

        if not changed_files:
            logger.info("No new/changed files in %s.", directory)
            return combined

        for file_path in changed_files:
            logger.info("Ingesting new/changed file: %s", file_path.name)
            result = self.ingest_file(file_path)
            combined = combined.merge(result)

        return combined

    def _index_batch(self, docs: list[Document], file_path: Path) -> IndexingResult:
        """Push documents to PGVector in batches and return a result summary."""
        combined = IndexingResult()
        for i in range(0, len(docs), self.batch_size):
            batch = docs[i : i + self.batch_size]
            try:
                raw = index(
                    docs_source=batch,
                    record_manager=self.record_manager,
                    vector_store=self.vector_store_sync,
                    cleanup="incremental",
                    source_id_key="source",
                    key_encoder="sha256",
                    batch_size=self.batch_size,
                )
                combined = combined.merge(
                    IndexingResult(
                        num_added=raw.get("num_added", 0),
                        num_updated=raw.get("num_updated", 0),
                        num_skipped=raw.get("num_skipped", 0),
                        num_deleted=raw.get("num_deleted", 0),
                    )
                )
            except Exception as exc:
                logger.error("Indexing batch failed for %s: %s", file_path.name, exc)
                combined.errors.append(str(exc))
        return combined

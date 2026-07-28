"""FredApiSource — fetches live series observations from the FRED REST API.

FRED API docs: https://fred.stlouisfed.org/docs/api/fred/series_observations.html

If no FRED_API_KEY is set the source raises a clear error rather than silently
returning stale data.  The orchestrator's system prompt instructs the agent to
fall back to ``search_pgvector`` when this source is unavailable.
"""

import logging
from datetime import datetime, timezone
from typing import Any

import httpx
from langchain_classic.indexes import SQLRecordManager
from langchain_core.documents import Document
from langchain_core.indexing import index

from .base import BaseSource, IndexingResult

logger = logging.getLogger(__name__)

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Series that are always kept fresh (used by ingest_all / scheduled refresh)
DEFAULT_SERIES = ["FEDFUNDS", "DFF", "CPILFESL", "DGS2"]


class FredApiSource(BaseSource):
    """
    On-demand and scheduled puller for FRED economic series.

    ``fetch(series_id)`` pulls the most recent observations for a single series
    and indexes them immediately so the result is cached in PGVector.

    ``ingest_all()`` refreshes all series listed in ``DEFAULT_SERIES``.
    """

    source_name = "fred_api"

    def __init__(
        self,
        api_key: str,
        vector_store_sync,
        record_manager: SQLRecordManager,
        observation_limit: int = 100,
    ):
        if not api_key:
            raise ValueError(
                "FRED_API_KEY is required for FredApiSource. "
                "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        self.api_key = api_key
        self.vector_store_sync = vector_store_sync
        self.record_manager = record_manager
        self.observation_limit = observation_limit

    def fetch(self, query: str, **kwargs: Any) -> list[Document]:
        """
        Pull live observations for a FRED series.
        ``query`` is the series ID (e.g. "FEDFUNDS", "UNRATE").
        """
        series_id = query.strip().upper()
        logger.info("FredApiSource: fetching series %s", series_id)
        docs = self._pull_series(series_id)
        if docs:
            self._index_docs(docs, series_id)
            logger.info(
                "FredApiSource: indexed %d observations for %s", len(docs), series_id
            )
        return docs

    def ingest_all(self) -> IndexingResult:
        """Refresh all default FRED series."""
        combined = IndexingResult()
        for series_id in DEFAULT_SERIES:
            try:
                docs = self._pull_series(series_id)
                if docs:
                    result = self._index_docs(docs, series_id)
                    combined = combined.merge(result)
            except Exception as exc:
                logger.error("Failed to refresh FRED series %s: %s", series_id, exc)
                combined.errors.append(f"{series_id}: {exc}")
        logger.info("FRED API ingest_all complete: %s", combined)
        return combined

    def _pull_series(self, series_id: str) -> list[Document]:
        """Call the FRED API and return Documents for each observation."""
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": self.observation_limit,
        }
        try:
            response = httpx.get(FRED_BASE_URL, params=params, timeout=15.0)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "FRED API HTTP error for %s: %s", series_id, exc.response.text
            )
            return []
        except Exception as exc:
            logger.error("FRED API request failed for %s: %s", series_id, exc)
            return []

        observations = data.get("observations", [])
        fetched_at = datetime.now(timezone.utc).isoformat()
        docs: list[Document] = []

        for obs in observations:
            date = obs.get("date", "").strip()
            value = obs.get("value", "").strip()
            # FRED uses "." to represent missing values
            if not date or value in ("", "."):
                continue
            source_key = f"fred_api:{series_id}:{date}"
            docs.append(
                Document(
                    page_content=(
                        f"As of {date}, the value for macroeconomic "
                        f"indicator {series_id} is {value} (source: FRED API)."
                    ),
                    metadata={
                        "source": source_key,
                        "metric": series_id,
                        "date": date,
                        "year": int(date.split("-")[0]) if "-" in date else 0,
                        "value": value,
                        "fetched_at": fetched_at,
                        "file_type": "fred_api",
                    },
                )
            )
        return docs

    def _index_docs(self, docs: list[Document], series_id: str) -> IndexingResult:
        try:
            raw = index(
                docs_source=docs,
                record_manager=self.record_manager,
                vector_store=self.vector_store_sync,
                cleanup="incremental",
                source_id_key="source",
                key_encoder="sha256",
            )
            return IndexingResult(
                num_added=raw.get("num_added", 0),
                num_updated=raw.get("num_updated", 0),
                num_skipped=raw.get("num_skipped", 0),
                num_deleted=raw.get("num_deleted", 0),
            )
        except Exception as exc:
            logger.error("Indexing failed for series %s: %s", series_id, exc)
            return IndexingResult(errors=[str(exc)])

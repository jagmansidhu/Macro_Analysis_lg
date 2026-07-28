"""WebSource — fetches and indexes arbitrary web pages for the RAG researcher.

Two-layer strategy:
  1. Firecrawl (if ``FIRECRAWL_API_KEY`` is set) — clean markdown via Firecrawl API.
  2. Fallback: plain ``httpx`` GET + BeautifulSoup HTML → text strip.

All fetched content is indexed into PGVector immediately so repeat queries are
served from the DB rather than hitting the network again.
"""

import logging
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from langchain_classic.indexes import SQLRecordManager
from langchain_core.documents import Document
from langchain_core.indexing import index

from .base import BaseSource, IndexingResult

logger = logging.getLogger(__name__)

# Maximum characters to index from a single web page
_MAX_CONTENT_CHARS = 12_000


def _domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc
    except Exception:
        return url


def _fetch_with_httpx(url: str) -> str:
    """Plain HTTP GET + BeautifulSoup fallback — no JS rendering."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; MacroResearchBot/1.0; "
            "+https://github.com/your-org/macro-analysis)"
        )
    }
    response = httpx.get(url, headers=headers, timeout=20.0, follow_redirects=True)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    # Remove nav / script / style noise
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def _fetch_with_firecrawl(url: str, api_key: str) -> str:
    """Use Firecrawl to scrape a URL and return clean markdown."""
    try:
        from firecrawl import FirecrawlApp  # lazy import

        app = FirecrawlApp(api_key=api_key)
        # Firecrawl v2: scrape() returns a Pydantic Document (not a dict)
        result = app.scrape(url, formats=["markdown"])
        return result.markdown or result.html or ""
    except ImportError:
        logger.warning("firecrawl-py not installed — falling back to httpx.")
        return _fetch_with_httpx(url)
    except Exception as exc:
        logger.warning("Firecrawl failed for %s (%s) — falling back to httpx.", url, exc)
        return _fetch_with_httpx(url)


class WebSource(BaseSource):
    """
    Fetches and indexes web pages on demand.

    ``fetch(url)`` — scrapes a URL, wraps result in a Document, indexes it.
    ``ingest_all()`` — no-op (web sources are purely on-demand).

    The ``source`` metadata key is the URL, so the same page won't be
    re-fetched if it's already in PGVector (``cleanup="incremental"`` deduplicates
    by source key + content hash).
    """

    source_name = "web"

    def __init__(
        self,
        vector_store_sync,
        record_manager: SQLRecordManager,
        firecrawl_api_key: str = "",
    ):
        self.vector_store_sync = vector_store_sync
        self.record_manager = record_manager
        self.firecrawl_api_key = firecrawl_api_key

    def fetch(self, query: str, **kwargs: Any) -> list[Document]:
        """
        Fetch a web page by URL (``query``) and index it.
        Returns a list containing the single indexed Document.
        """
        url = query.strip()
        logger.info("WebSource: fetching %s", url)

        content = self._scrape(url)
        if not content or not content.strip():
            logger.warning("WebSource: no content extracted from %s", url)
            return []

        content = content[:_MAX_CONTENT_CHARS]
        fetched_at = datetime.now(timezone.utc).isoformat()

        doc = Document(
            page_content=content,
            metadata={
                "source": url,
                "url": url,
                "domain": _domain_from_url(url),
                "fetched_at": fetched_at,
                "file_type": "web",
            },
        )

        result = self._index_docs([doc])
        logger.info(
            "WebSource: indexed page %s — %s", _domain_from_url(url), result
        )
        return [doc]

    def ingest_all(self) -> IndexingResult:
        """No-op — web content is always fetched on demand."""
        logger.debug("WebSource.ingest_all called — nothing to do (on-demand only).")
        return IndexingResult()

    def _scrape(self, url: str) -> str:
        """Try Firecrawl first; fall back to httpx + BeautifulSoup."""
        if self.firecrawl_api_key:
            return _fetch_with_firecrawl(url, self.firecrawl_api_key)
        return _fetch_with_httpx(url)

    def _index_docs(self, docs: list[Document]) -> IndexingResult:
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
            logger.error("WebSource: indexing failed: %s", exc)
            return IndexingResult(errors=[str(exc)])

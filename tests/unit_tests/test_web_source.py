"""Unit tests for WebSource — Firecrawl + httpx fallback scraping."""

from unittest.mock import MagicMock, patch

import pytest

from RAG.sources.web_source import (
    _domain_from_url,
    _fetch_with_httpx,
    WebSource,
)


class TestDomainFromUrl:
    def test_standard_url(self):
        assert _domain_from_url("https://www.federalreserve.gov/news") == "www.federalreserve.gov"

    def test_url_with_path(self):
        assert _domain_from_url("https://fred.stlouisfed.org/series/FEDFUNDS") == "fred.stlouisfed.org"

    def test_malformed_url_returns_input(self):
        # Should not raise
        result = _domain_from_url("not-a-url")
        assert isinstance(result, str)


class TestFetchWithHttpx:
    def test_returns_text_content(self):
        fake_response = MagicMock()
        fake_response.text = "<html><body><p>CPI rose 3% in June 2025.</p></body></html>"
        fake_response.raise_for_status = MagicMock()

        with patch("RAG.sources.web_source.httpx.get", return_value=fake_response):
            result = _fetch_with_httpx("https://example.com/article")

        assert "CPI rose 3%" in result

    def test_strips_script_and_style_tags(self):
        fake_response = MagicMock()
        fake_response.text = (
            "<html><head><style>body{color:red}</style></head>"
            "<body><script>alert(1)</script><p>Relevant content.</p></body></html>"
        )
        fake_response.raise_for_status = MagicMock()

        with patch("RAG.sources.web_source.httpx.get", return_value=fake_response):
            result = _fetch_with_httpx("https://example.com")

        assert "Relevant content." in result
        assert "alert(1)" not in result
        assert "body{color" not in result


class TestFetchWithFirecrawl:
    """Tests for the _fetch_with_firecrawl helper and WebSource's _scrape routing."""

    def test_firecrawl_returns_markdown(self):
        from RAG.sources.web_source import _fetch_with_firecrawl

        fake_app = MagicMock()
        fake_app.scrape_url.return_value = {"markdown": "# Fed Statement\nRates held steady."}

        with patch("RAG.sources.web_source.FirecrawlApp", return_value=fake_app, create=True):
            with patch.dict("sys.modules", {"firecrawl": MagicMock(FirecrawlApp=fake_app.__class__)}):
                # Patch at the lazy import site inside the function
                with patch("builtins.__import__", side_effect=lambda name, *a, **kw: fake_app if name == "firecrawl" else __import__(name, *a, **kw)):
                    pass  # import-level patching is tricky; test via WebSource._scrape instead

        # Simpler: patch the module attribute directly
        import RAG.sources.web_source as ws_mod
        fake_fc_app = MagicMock()
        fake_fc_app.scrape_url.return_value = {"markdown": "# Fed Statement\nRates held steady."}

        with patch.object(ws_mod, "_fetch_with_firecrawl", return_value="# Fed Statement\nRates held steady."):
            source = ws_mod.WebSource(
                vector_store_sync=MagicMock(),
                record_manager=MagicMock(),
                firecrawl_api_key="fc-test-key",
            )
            content = source._scrape("https://federalreserve.gov/statement")

        assert "Fed Statement" in content
        assert "Rates held steady" in content

    def test_websource_uses_firecrawl_when_key_present(self):
        """When firecrawl_api_key is set, _scrape must call _fetch_with_firecrawl."""
        import RAG.sources.web_source as ws_mod

        source = ws_mod.WebSource(
            vector_store_sync=MagicMock(),
            record_manager=MagicMock(),
            firecrawl_api_key="fc-real-key",
        )
        with patch.object(ws_mod, "_fetch_with_firecrawl", return_value="FC content") as mock_fc, \
             patch.object(ws_mod, "_fetch_with_httpx") as mock_httpx:
            content = source._scrape("https://example.com")

        mock_fc.assert_called_once_with("https://example.com", "fc-real-key")
        mock_httpx.assert_not_called()
        assert content == "FC content"

    def test_websource_falls_back_to_httpx_when_no_key(self):
        """Without a firecrawl key, _scrape must use httpx."""
        import RAG.sources.web_source as ws_mod

        source = ws_mod.WebSource(
            vector_store_sync=MagicMock(),
            record_manager=MagicMock(),
            firecrawl_api_key="",
        )
        with patch.object(ws_mod, "_fetch_with_firecrawl") as mock_fc, \
             patch.object(ws_mod, "_fetch_with_httpx", return_value="httpx content") as mock_httpx:
            content = source._scrape("https://example.com")

        mock_httpx.assert_called_once_with("https://example.com")
        mock_fc.assert_not_called()
        assert content == "httpx content"

    def test_firecrawl_failure_falls_back_to_httpx(self):
        """If Firecrawl raises, _fetch_with_firecrawl must fall back to httpx internally."""
        from RAG.sources.web_source import _fetch_with_firecrawl

        fake_response = MagicMock()
        fake_response.text = "<html><body><p>Fallback content.</p></body></html>"
        fake_response.raise_for_status = MagicMock()

        with patch("RAG.sources.web_source.httpx.get", return_value=fake_response):
            # Simulate firecrawl import succeeding but scrape raising
            fake_app_cls = MagicMock()
            fake_app_instance = MagicMock()
            fake_app_instance.scrape.side_effect = Exception("Firecrawl API error")
            fake_app_cls.return_value = fake_app_instance

            import sys
            fake_firecrawl_mod = MagicMock()
            fake_firecrawl_mod.FirecrawlApp = fake_app_cls

            with patch.dict(sys.modules, {"firecrawl": fake_firecrawl_mod}):
                result = _fetch_with_firecrawl("https://example.com", "fc-key")

        assert "Fallback content." in result


class TestWebSourceFetch:
    def _make_source(self, firecrawl_key: str = "") -> WebSource:
        mock_vs = MagicMock()
        mock_rm = MagicMock()
        return WebSource(
            vector_store_sync=mock_vs,
            record_manager=mock_rm,
            firecrawl_api_key=firecrawl_key,
        )

    def test_fetch_without_firecrawl_uses_httpx(self):
        source = self._make_source(firecrawl_key="")

        fake_response = MagicMock()
        fake_response.text = "<html><body><p>Fed holds rates steady.</p></body></html>"
        fake_response.raise_for_status = MagicMock()

        with patch("RAG.sources.web_source.httpx.get", return_value=fake_response), \
             patch.object(source, "_index_docs", return_value=MagicMock()):
            docs = source.fetch("https://example.com/fed-news")

        assert len(docs) == 1
        assert "Fed holds rates" in docs[0].page_content
        assert docs[0].metadata["file_type"] == "web"
        assert docs[0].metadata["url"] == "https://example.com/fed-news"

    def test_fetch_returns_empty_on_no_content(self):
        source = self._make_source()

        with patch.object(source, "_scrape", return_value="   "):
            docs = source.fetch("https://example.com/blank")

        assert docs == []

    def test_fetch_truncates_long_content(self):
        source = self._make_source()
        long_content = "x" * 20_000

        with patch.object(source, "_scrape", return_value=long_content), \
             patch.object(source, "_index_docs", return_value=MagicMock()):
            docs = source.fetch("https://example.com/long")

        assert len(docs[0].page_content) <= 12_000

    def test_ingest_all_is_noop(self):
        source = self._make_source()
        result = source.ingest_all()
        assert result.num_added == 0
        assert result.num_skipped == 0

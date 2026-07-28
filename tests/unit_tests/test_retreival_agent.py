"""Unit tests for retreival_agent."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

# Mock config before importing retreival_agent
import sys
mock_config = MagicMock()
mock_config.FRED_API_KEY = "test"
mock_config.FIRECRAWL_API_KEY = "test"
mock_config.TAVILY_API_KEY = "test"
mock_config.llm = MagicMock()
mock_config.vector_store = MagicMock()
mock_config.vector_store.asimilarity_search = AsyncMock()
mock_config.vector_store_sync = MagicMock()
mock_config.record_manager = MagicMock()
sys.modules["config"] = mock_config

from RAG.retreival_agent import (
    get_latest_data,
    fetch_fred_api,
    fetch_web_page,
    search_web_news,
    retrieval_agent,
    system_prompt,
)

pytestmark = pytest.mark.anyio


class TestGetLatestData:
    @patch("RAG.retreival_agent.vector_store")
    async def test_found_in_first_search(self, mock_vs):
        mock_vs.asimilarity_search = AsyncMock(return_value=[
            MagicMock(page_content="data1", metadata={"date": "2025-01-01"}),
            MagicMock(page_content="data2", metadata={"date": "2025-02-01"})
        ])
        
        result = await get_latest_data.ainvoke({"metric": "FEDFUNDS", "n": 1})
        assert "data2" in result
        assert "data1" not in result

    @patch("RAG.retreival_agent.vector_store")
    async def test_not_found_returns_message(self, mock_vs):
        mock_vs.asimilarity_search = AsyncMock(return_value=[])
        result = await get_latest_data.ainvoke({"metric": "UNKNOWN", "n": 1})
        assert "No data found" in result


class TestFetchFredApi:
    @patch("RAG.retreival_agent._fred_source")
    def test_fetch_success(self, mock_fred):
        mock_fred.fetch.return_value = [
            MagicMock(page_content="data1", metadata={"date": "2025-01-01"}),
            MagicMock(page_content="data2", metadata={"date": "2025-02-01"})
        ]
        result = fetch_fred_api.invoke({"series_id": "FEDFUNDS", "limit": 1})
        assert "data2" in result
        assert "data1" not in result

    @patch("RAG.retreival_agent._fred_source")
    def test_fetch_no_data(self, mock_fred):
        mock_fred.fetch.return_value = []
        result = fetch_fred_api.invoke({"series_id": "UNKNOWN"})
        assert "No data returned" in result


class TestFetchWebPage:
    @patch("RAG.retreival_agent._web_source")
    def test_fetch_success(self, mock_web):
        mock_web.fetch.return_value = [MagicMock(page_content="content")]
        result = fetch_web_page.invoke({"url": "http://example.com"})
        assert result == "content"

    @patch("RAG.retreival_agent._web_source")
    def test_fetch_no_data(self, mock_web):
        mock_web.fetch.return_value = []
        result = fetch_web_page.invoke({"url": "http://example.com"})
        assert "No content could be extracted" in result


class TestSearchWebNews:
    @patch("tavily.TavilyClient")
    @patch("RAG.retreival_agent._web_source")
    def test_search_success(self, mock_web, mock_tavily_cls):
        mock_client = MagicMock()
        mock_tavily_cls.return_value = mock_client
        mock_client.search.return_value = {
            "answer": "A summary",
            "results": [
                {"url": "http://a.com", "title": "A", "content": "Content A"}
            ]
        }
        
        result = search_web_news.invoke({"topic": "Fed", "max_results": 1})
        assert "A summary" in result
        assert "Content A" in result
        mock_web.fetch.assert_called_once_with("http://a.com")

    @patch("tavily.TavilyClient")
    def test_search_no_results(self, mock_tavily_cls):
        mock_client = MagicMock()
        mock_tavily_cls.return_value = mock_client
        mock_client.search.return_value = {"results": []}
        
        result = search_web_news.invoke({"topic": "Fed"})
        assert "No web results found" in result


def test_agent_created():
    assert retrieval_agent is not None
    assert "You are a macroeconomic researcher agent" in system_prompt

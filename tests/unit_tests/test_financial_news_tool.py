"""Unit tests for the search_financial_news tool.

All Tavily calls are mocked — no real network requests.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tavily_response(results: list[dict], answer: str = "") -> dict:
    return {"results": results, "answer": answer}


SAMPLE_RESULTS = [
    {
        "url": "https://www.cnbc.com/2025/07/28/fed-rate-cut.html",
        "title": "Fed signals rate cut in September, analysts say",
        "content": "Wall Street analysts broadly expect the Fed to cut rates in September...",
        "source": "cnbc.com",
    },
    {
        "url": "https://www.bloomberg.com/2025/07/28/sp500-target.html",
        "title": "Goldman raises S&P 500 year-end target to 6000",
        "content": "Goldman Sachs lifted its S&P 500 target citing strong earnings growth...",
        "source": "bloomberg.com",
    },
]


def _patch_tavily(results=SAMPLE_RESULTS, answer="Markets optimistic on rate cuts."):
    fake_client = MagicMock()
    fake_client.search.return_value = _make_tavily_response(results, answer)
    fake_module = MagicMock()
    fake_module.TavilyClient.return_value = fake_client
    return fake_module, fake_client


# ---------------------------------------------------------------------------
# Import the tool after mocking config so the module loads cleanly
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_config(monkeypatch):
    """Prevent real DB/API connections when importing retreival_agent."""
    import types

    fake_config = types.ModuleType("config")
    fake_config.llm = MagicMock()
    fake_config.vector_store = MagicMock()
    fake_config.vector_store_sync = MagicMock()
    fake_config.record_manager = MagicMock()
    fake_config.FRED_API_KEY = "test-fred"
    fake_config.FIRECRAWL_API_KEY = ""
    fake_config.TAVILY_API_KEY = "test-tavily"
    fake_config.WATCH_DIRS = []

    # Patch the whole config module before the agent is imported
    monkeypatch.setitem(sys.modules, "config", fake_config)

    # Also stub out langchain imports that need DB
    for mod in [
        "langchain.agents",
        "langchain_core.tools.retriever",
    ]:
        if mod not in sys.modules:
            monkeypatch.setitem(sys.modules, mod, MagicMock())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSearchFinancialNewsDefaults:
    def test_uses_default_financial_domains_when_no_override(self):
        from RAG.retreival_agent import search_financial_news, DEFAULT_FINANCIAL_DOMAINS

        fake_tavily, fake_client = _patch_tavily()
        with patch.dict(sys.modules, {"tavily": fake_tavily}), \
             patch("RAG.retreival_agent._web_source") as mock_ws:
            mock_ws.fetch.return_value = []
            result = search_financial_news.invoke({
                "topic": "Fed rate cut analyst views",
            })

        call_kwargs = fake_client.search.call_args.kwargs
        assert call_kwargs["include_domains"] == DEFAULT_FINANCIAL_DOMAINS
        assert call_kwargs["topic"] == "finance"

    def test_uses_override_domains_when_provided(self):
        from RAG.retreival_agent import search_financial_news

        fake_tavily, fake_client = _patch_tavily()
        custom = ["wowa.ca", "ratehub.ca"]
        with patch.dict(sys.modules, {"tavily": fake_tavily}), \
             patch("RAG.retreival_agent._web_source") as mock_ws:
            mock_ws.fetch.return_value = []
            result = search_financial_news.invoke({
                "topic": "Canadian mortgage rates 2025",
                "domains": custom,
            })

        call_kwargs = fake_client.search.call_args.kwargs
        assert call_kwargs["include_domains"] == custom

    def test_no_domain_filter_when_empty_list(self):
        """Passing domains=[] should omit include_domains from the Tavily call."""
        from RAG.retreival_agent import search_financial_news

        fake_tavily, fake_client = _patch_tavily()
        with patch.dict(sys.modules, {"tavily": fake_tavily}), \
             patch("RAG.retreival_agent._web_source") as mock_ws:
            mock_ws.fetch.return_value = []
            result = search_financial_news.invoke({
                "topic": "global macro trends",
                "domains": [],
            })

        call_kwargs = fake_client.search.call_args.kwargs
        assert "include_domains" not in call_kwargs

    def test_time_range_passed_through(self):
        from RAG.retreival_agent import search_financial_news

        fake_tavily, fake_client = _patch_tavily()
        with patch.dict(sys.modules, {"tavily": fake_tavily}), \
             patch("RAG.retreival_agent._web_source"):
            search_financial_news.invoke({
                "topic": "CPI reaction",
                "time_range": "day",
            })

        call_kwargs = fake_client.search.call_args.kwargs
        assert call_kwargs["time_range"] == "day"


class TestSearchFinancialNewsOutput:
    def test_includes_market_summary_when_answer_present(self):
        from RAG.retreival_agent import search_financial_news

        fake_tavily, _ = _patch_tavily(answer="Analysts broadly bullish on equities.")
        with patch.dict(sys.modules, {"tavily": fake_tavily}), \
             patch("RAG.retreival_agent._web_source"):
            result = search_financial_news.invoke({"topic": "S&P 500 outlook"})

        assert "Market Summary" in result
        assert "Analysts broadly bullish" in result

    def test_article_titles_and_urls_in_output(self):
        from RAG.retreival_agent import search_financial_news

        fake_tavily, _ = _patch_tavily()
        with patch.dict(sys.modules, {"tavily": fake_tavily}), \
             patch("RAG.retreival_agent._web_source"):
            result = search_financial_news.invoke({"topic": "Fed rate cut"})

        assert "Fed signals rate cut" in result
        assert "cnbc.com" in result
        assert "Goldman raises S&P 500" in result

    def test_sentiment_block_appended(self):
        from RAG.retreival_agent import search_financial_news

        fake_tavily, _ = _patch_tavily()
        with patch.dict(sys.modules, {"tavily": fake_tavily}), \
             patch("RAG.retreival_agent._web_source"):
            result = search_financial_news.invoke({"topic": "market sentiment"})

        assert "Analyst Sentiment Signals" in result
        assert "Bullish | Bearish | Mixed" in result

    def test_each_result_url_indexed(self):
        from RAG.retreival_agent import search_financial_news

        fake_tavily, _ = _patch_tavily()
        with patch.dict(sys.modules, {"tavily": fake_tavily}), \
             patch("RAG.retreival_agent._web_source") as mock_ws:
            mock_ws.fetch.return_value = []
            search_financial_news.invoke({"topic": "rate cut"})

        fetched_urls = [call.args[0] for call in mock_ws.fetch.call_args_list]
        assert "https://www.cnbc.com/2025/07/28/fed-rate-cut.html" in fetched_urls
        assert "https://www.bloomberg.com/2025/07/28/sp500-target.html" in fetched_urls


class TestSearchFinancialNewsEdgeCases:
    def test_empty_results_returns_helpful_message(self):
        from RAG.retreival_agent import search_financial_news

        fake_tavily, _ = _patch_tavily(results=[], answer="")
        with patch.dict(sys.modules, {"tavily": fake_tavily}), \
             patch("RAG.retreival_agent._web_source"):
            result = search_financial_news.invoke({"topic": "obscure topic"})

        assert "No financial news found" in result

    def test_missing_tavily_key_returns_clear_error(self, monkeypatch):
        import RAG.retreival_agent as agent_mod
        monkeypatch.setattr(agent_mod, "TAVILY_API_KEY", "")

        result = agent_mod.search_financial_news.invoke({"topic": "Fed outlook"})
        assert "TAVILY_API_KEY" in result

    def test_indexing_failure_is_non_fatal(self):
        from RAG.retreival_agent import search_financial_news

        fake_tavily, _ = _patch_tavily()
        with patch.dict(sys.modules, {"tavily": fake_tavily}), \
             patch("RAG.retreival_agent._web_source") as mock_ws:
            mock_ws.fetch.side_effect = Exception("DB down")
            # Should not raise — indexing errors are swallowed
            result = search_financial_news.invoke({"topic": "Fed rate cut"})

        assert "Fed signals rate cut" in result  # content still returned

    def test_tavily_exception_returns_error_string(self):
        from RAG.retreival_agent import search_financial_news

        fake_module = MagicMock()
        fake_module.TavilyClient.side_effect = Exception("API failure")
        with patch.dict(sys.modules, {"tavily": fake_module}), \
             patch("RAG.retreival_agent._web_source"):
            result = search_financial_news.invoke({"topic": "GDP growth"})

        assert "Financial news search failed" in result


class TestDefaultDomainList:
    def test_default_domains_include_major_outlets(self):
        from RAG.retreival_agent import DEFAULT_FINANCIAL_DOMAINS

        required = {"cnbc.com", "bloomberg.com", "reuters.com", "ft.com", "wsj.com"}
        assert required.issubset(set(DEFAULT_FINANCIAL_DOMAINS)), (
            f"Missing required outlets: {required - set(DEFAULT_FINANCIAL_DOMAINS)}"
        )

    def test_default_domains_are_lowercase(self):
        from RAG.retreival_agent import DEFAULT_FINANCIAL_DOMAINS

        for domain in DEFAULT_FINANCIAL_DOMAINS:
            assert domain == domain.lower(), f"Domain not lowercase: {domain}"

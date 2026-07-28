"""API key smoke tests — verify that configured keys are valid and services respond.

Each test skips automatically if its key is not set in the environment.
These are NOT mocked — they make a real (but minimal) network request.

Run with:
    uv run pytest tests/integration_tests/test_api_keys_smoke.py -v

To run just one service:
    uv run pytest tests/integration_tests/test_api_keys_smoke.py -v -k fred
    uv run pytest tests/integration_tests/test_api_keys_smoke.py -v -k firecrawl
    uv run pytest tests/integration_tests/test_api_keys_smoke.py -v -k tavily
"""

import os

import httpx
import pytest

def _require_key(env_var: str) -> str:
    """Skip the test if the key is missing or still the placeholder value."""
    key = os.getenv(env_var, "").strip()
    if not key or key.startswith("your-"):
        pytest.skip(f"{env_var} is not configured — skipping smoke test.")
    return key
class TestFredApiKey:
    """
    Hits the FRED /series/observations endpoint for a tiny, stable series (DFF).
    A 200 response with observations confirms the key is valid.
    """

    def test_fred_key_is_valid(self):
        api_key = _require_key("FRED_API_KEY")

        response = httpx.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": "DFF",
                "api_key": api_key,
                "file_type": "json",
                "limit": 1,
                "sort_order": "desc",
            },
            timeout=15.0,
        )

        assert response.status_code == 200, (
            f"FRED API returned {response.status_code}: {response.text[:200]}"
        )
        data = response.json()
        assert "observations" in data, f"Unexpected response shape: {data}"
        assert len(data["observations"]) >= 1, "FRED returned zero observations for DFF"

    def test_fred_key_is_rejected_when_bad(self):
        """
        Sanity check: FRED returns 400 / error for a bad key.
        Skipped unless FRED_API_KEY is present so we can verify the contrast.
        """
        _require_key("FRED_API_KEY")

        response = httpx.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": "DFF",
                "api_key": "bad-key-000",
                "file_type": "json",
                "limit": 1,
            },
            timeout=15.0,
        )
        assert response.status_code == 400, (
            "Expected 400 for a bad FRED key — API behaviour may have changed."
        )
class TestFirecrawlApiKey:
    """
    Scrapes a stable, short public URL (example.com) using the Firecrawl API.
    A non-empty markdown response confirms the key is valid.
    """

    def test_firecrawl_key_is_valid(self):
        api_key = _require_key("FIRECRAWL_API_KEY")

        try:
            from firecrawl import FirecrawlApp
        except ImportError:
            pytest.skip("firecrawl-py not installed — run: uv add firecrawl-py")

        app = FirecrawlApp(api_key=api_key)
        result = app.scrape(
            "https://example.com",
            formats=["markdown"],
        )

        assert result, "Firecrawl returned an empty result"
        content = result.markdown or result.html or ""
        assert content.strip(), (
            "Firecrawl returned a result but with no markdown/html content — "
            f"full response: {result}"
        )

    def test_firecrawl_key_rejected_when_bad(self):
        """Firecrawl returns 401 for an invalid key."""
        _require_key("FIRECRAWL_API_KEY")

        response = httpx.post(
            "https://api.firecrawl.dev/v1/scrape",
            json={"url": "https://example.com", "formats": ["markdown"]},
            headers={"Authorization": "Bearer bad-key-000"},
            timeout=15.0,
        )
        assert response.status_code in (401, 403), (
            f"Expected 401/403 for a bad Firecrawl key, got {response.status_code}"
        )
class TestTavilyApiKey:
    """
    Runs a minimal Tavily search for a static, well-indexed topic.
    A non-empty results list confirms the key is valid.
    """

    def test_tavily_key_is_valid(self):
        api_key = _require_key("TAVILY_API_KEY")

        try:
            from tavily import TavilyClient
        except ImportError:
            pytest.skip("tavily-python not installed — run: uv add tavily-python")

        client = TavilyClient(api_key=api_key)
        response = client.search(
            query="Federal Reserve interest rate",
            search_depth="basic",
            max_results=1,
        )

        assert response, "Tavily returned an empty response"
        results = response.get("results", [])
        assert len(results) >= 1, (
            f"Tavily returned zero results — full response: {response}"
        )
        first = results[0]
        assert "url" in first, f"Result missing 'url': {first}"
        assert "content" in first, f"Result missing 'content': {first}"

    def test_tavily_key_rejected_when_bad(self):
        """Tavily returns 401 for an invalid API key."""
        _require_key("TAVILY_API_KEY")

        response = httpx.post(
            "https://api.tavily.com/search",
            json={"query": "test", "api_key": "tvly-bad-key-000"},
            timeout=15.0,
        )
        assert response.status_code in (401, 403), (
            f"Expected 401/403 for a bad Tavily key, got {response.status_code}"
        )
class TestGeminiApiKey:
    """
    Calls the Gemini embeddings endpoint with a short string.
    Confirms the key is valid and the embedding dimension is 1536 (as required).
    """

    def test_gemini_key_is_valid_and_embedding_dim_is_1536(self):
        api_key = _require_key("GEMINI_API_KEY")

        try:
            import google.generativeai as genai
        except ImportError:
            pytest.skip("google-generativeai not installed")

        genai.configure(api_key=api_key)
        result = genai.embed_content(
            model="models/text-embedding-004",
            content="Federal Reserve funds rate",
        )

        embedding = result.get("embedding") or result["embedding"]
        assert embedding, "Gemini returned an empty embedding"
        assert len(embedding) == 1536, (
            f"Embedding dimension mismatch: expected 1536, got {len(embedding)}. "
            "See EMBEDDING_DIM in config.py — never change this without dropping the DB tables."
        )

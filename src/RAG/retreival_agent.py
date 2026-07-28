"""RAG Orchestrator — the researcher brain.

Exposes a six-tool ReAct agent to any analysis agent that needs information:

  1. search_pgvector        — semantic search over all indexed content (always first)
  2. get_latest_data        — most-recent N rows for a known metric, sorted by date
  3. fetch_fred_api         — pull live FRED series data and index it
  4. fetch_web_page         — scrape a URL via Firecrawl / httpx and index it
  5. search_web_news        — broad web news search for macro topics (Tavily)
  6. search_financial_news  — domain-filtered financial news + analyst sentiment

Routing rules (embedded in system prompt):
  - Always try search_pgvector first (fast, free, cached).
  - For "latest" / "current" data: use get_latest_data.
  - For a FRED series not in the DB or if DB results are stale: use fetch_fred_api.
  - For news articles, reports, or arbitrary URLs: use fetch_web_page.
  - For broad macro news queries: use search_web_news.
  - For analyst views, market sentiment, or outlet-specific financial coverage:
    use search_financial_news (defaults to CNBC/Bloomberg/Reuters/FT/WSJ;
    override domains for topic-specific sources e.g. WOWA for Canadian mortgages).
"""

import asyncio
import logging

from config import llm, vector_store, vector_store_sync, record_manager
from config import FRED_API_KEY, FIRECRAWL_API_KEY, TAVILY_API_KEY
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.tools.retriever import create_retriever_tool

from RAG.sources.local_file_source import LocalFileSource
from RAG.sources.fred_api_source import FredApiSource
from RAG.sources.web_source import WebSource
from RAG.prompts import RETRIEVAL_AGENT_PROMPT

logger = logging.getLogger(__name__)

_web_source = WebSource(
    vector_store_sync=vector_store_sync,
    record_manager=record_manager,
    firecrawl_api_key=FIRECRAWL_API_KEY,
)

_fred_source: FredApiSource | None = None
if FRED_API_KEY:
    try:
        _fred_source = FredApiSource(
            api_key=FRED_API_KEY,
            vector_store_sync=vector_store_sync,
            record_manager=record_manager,
        )
    except ValueError as e:
        logger.warning("FredApiSource unavailable: %s", e)

retriever = vector_store.as_retriever(search_kwargs={"k": 20})

search_pgvector = create_retriever_tool(
    retriever,
    name="search_pgvector",
    description=(
        "Search all indexed research content using semantic similarity. "
        "Covers FRED CSV data, research PDFs, web pages, and API-fetched series. "
        "ALWAYS call this tool first before reaching for any live-fetch tool. "
        "Good for questions like 'What happened to CPI in 2020?' or 'Fed funds rate trends'. "
        "NOT reliable for 'most recent' / 'latest' queries — use get_latest_data for those."
    ),
)

@tool
async def get_latest_data(metric: str, n: int = 5) -> str:
    """Get the N most recent data points for a specific metric, sorted by date descending.

    Use when the user asks for the 'most recent', 'latest', or 'current' value.

    Args:
        metric: Series ID to look up (e.g. 'CPILFESL', 'DFF', 'FEDFUNDS', 'DGS2').
                Available in DB: CPILFESL (Core CPI), DFF (Daily Fed Funds),
                FEDFUNDS (Monthly Fed Funds), DGS2 (2-Year Treasury Yield).
        n: Number of most recent data points to return (default 5).
    """
    results = await vector_store.asimilarity_search(
        query=f"most recent {metric} data",
        k=100,
        filter={"metric": metric},
    )

    if not results:
        results = await vector_store.asimilarity_search(
            query=f"most recent {metric} data",
            k=200,
        )
        results = [doc for doc in results if metric in doc.metadata.get("metric", "")]

    if not results:
        return f"No data found for metric '{metric}' in the database."

    sorted_results = sorted(
        results, key=lambda d: d.metadata.get("date", ""), reverse=True
    )
    top_n = sorted_results[:n]
    return "\n".join(doc.page_content for doc in top_n)


@tool
def fetch_fred_api(series_id: str, limit: int = 20) -> str:
    """Fetch live economic data for a FRED series and index it for future queries.

    Use when:
    - The user asks for a FRED series that may not be in the database yet.
    - The database results look stale and you need fresh observations.
    - The user explicitly asks for 'live' or 'real-time' FRED data.

    Args:
        series_id: FRED series identifier (e.g. 'UNRATE', 'GDP', 'CPIAUCSL').
                   Must be uppercase. See https://fred.stlouisfed.org for available series.
        limit: Number of most recent observations to return (default 20).
    """
    if _fred_source is None:
        return (
            "FRED API source is not configured (FRED_API_KEY is missing). "
            "Try search_pgvector for data already in the database."
        )
    docs = _fred_source.fetch(series_id)
    if not docs:
        return f"No data returned from FRED API for series '{series_id}'."
    # Return the most recent `limit` observations
    sorted_docs = sorted(
        docs, key=lambda d: d.metadata.get("date", ""), reverse=True
    )
    return "\n".join(doc.page_content for doc in sorted_docs[:limit])


@tool
def fetch_web_page(url: str) -> str:
    """Fetch a web page, extract its text content, and index it for future queries.

    Use when:
    - The user provides a specific URL (news article, Fed statement, report, whitepaper).
    - The analysis agent needs content from an external source not yet in the database.

    The page content is indexed immediately so subsequent calls for the same URL
    are served from the database.

    Args:
        url: Full URL to fetch (e.g. 'https://www.federalreserve.gov/newsevents/...')
    """
    docs = _web_source.fetch(url)
    if not docs:
        return f"No content could be extracted from '{url}'."
    return docs[0].page_content


@tool
def search_web_news(topic: str, max_results: int = 5) -> str:
    """Search the web for recent news and analysis on a macroeconomic topic using Tavily.

    Use when:
    - The user wants recent news, commentary, or analyst views on a macro topic.
    - The question is about something not captured in FRED data (e.g. policy rumours,
      central bank speeches, inflation commentary, market reactions).

    Results are indexed into the database so follow-up questions are fast.

    Args:
        topic: The macro topic to search (e.g. 'Fed rate cut expectations 2025',
               'CPI inflation July 2025', 'Treasury yield curve inversion').
        max_results: Number of search results to return (default 5).
    """
    if not TAVILY_API_KEY:
        return (
            "Web news search requires TAVILY_API_KEY. "
            "Sign up free at https://tavily.com. "
            "Use fetch_web_page with a specific URL as an alternative."
        )
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(
            query=topic,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True,  # Tavily's synthesised answer snippet
        )

        results = response.get("results", [])
        if not results:
            return f"No web results found for '{topic}'."

        snippets: list[str] = []

        # Include Tavily's synthesised answer if present
        answer = response.get("answer", "")
        if answer:
            snippets.append(f"**Summary**: {answer}")

        for item in results:
            url = item.get("url", "")
            title = item.get("title", "")
            content = item.get("content", "")[:600]
            if url and content:
                # Index the full page in the background for future retrieval
                try:
                    _web_source.fetch(url)
                except Exception:
                    pass  # Non-fatal — snippet is still returned
                snippets.append(f"**{title}**\n[{url}]\n{content}")

        return "\n\n---\n\n".join(snippets) if snippets else "No useful results found."

    except ImportError:
        return "tavily-python is not installed. Run: uv add tavily-python"
    except Exception as exc:
        logger.error("search_web_news failed for '%s': %s", topic, exc)
        return f"Web news search failed: {exc}"


# ---------------------------------------------------------------------------
# Tool 6 — Financial news search with analyst sentiment
# ---------------------------------------------------------------------------

# Default outlets strongly preferred for financial/macro topics.
# The agent can pass a custom list via the `domains` argument when a
# topic-specific source is more authoritative (e.g. "wowa.ca" for Canadian
# mortgage rates, "bis.org" for international settlements).
DEFAULT_FINANCIAL_DOMAINS: list[str] = [
    "cnbc.com",
    "bloomberg.com",
    "reuters.com",
    "ft.com",
    "wsj.com",
    "marketwatch.com",
    "barrons.com",
]


@tool
def search_financial_news(
    topic: str,
    time_range: str = "week",
    max_results: int = 5,
    domains: list[str] | None = None,
) -> str:
    """Search financial news outlets for a macro topic and extract analyst sentiment.

    By default restricts results to: CNBC, Bloomberg, Reuters, FT, WSJ,
    MarketWatch, Barron's. Pass a custom `domains` list to override for
    topic-specific sources (e.g. ["wowa.ca"] for Canadian mortgage data,
    ["bis.org"] for international banking settlements, ["bank-banque-canada.ca"]
    for Bank of Canada commentary).

    Returns article snippets from each outlet, then a synthesised analyst
    sentiment (Bullish / Bearish / Mixed) based on the retrieved content.

    Use when:
    - The user asks for analyst views, market sentiment, price targets, or
      upgrade/downgrade commentary.
    - The user specifically wants CNBC, Bloomberg, or Reuters coverage.
    - A regional or specialist source is more appropriate for the topic
      (specify it via the `domains` argument).

    Args:
        topic: The financial/macro topic (e.g. 'Fed rate cut Wall Street reaction',
               'S&P 500 analyst outlook', 'Canadian mortgage rate forecast wowa').
        time_range: Recency — 'day', 'week' (default), 'month', or 'year'.
        max_results: Number of articles to return (default 5).
        domains: Override the default outlet list. Pass an empty list to search
                 all domains (no filter). Pass specific domains for topic-specialist
                 sources. Defaults to the major financial news outlets.
    """
    if not TAVILY_API_KEY:
        return (
            "search_financial_news requires TAVILY_API_KEY. "
            "Sign up free at https://tavily.com. "
            "Use fetch_web_page with a specific URL as an alternative."
        )
    try:
        from tavily import TavilyClient

        active_domains = DEFAULT_FINANCIAL_DOMAINS if domains is None else domains

        client = TavilyClient(api_key=TAVILY_API_KEY)
        search_kwargs: dict = {
            "query": topic,
            "topic": "finance",
            "search_depth": "advanced",
            "time_range": time_range,
            "max_results": max_results,
            "include_answer": True,
        }
        if active_domains:
            search_kwargs["include_domains"] = active_domains

        response = client.search(**search_kwargs)

        results = response.get("results", [])
        if not results:
            domain_hint = f" in {active_domains}" if active_domains else ""
            return f"No financial news found for '{topic}'{domain_hint}."

        snippets: list[str] = []

        # Tavily's synthesised answer (finance-tuned)
        answer = response.get("answer", "")
        if answer:
            snippets.append(f"**Market Summary**: {answer}")

        raw_texts: list[str] = []
        for item in results:
            url = item.get("url", "")
            title = item.get("title", "")
            content = item.get("content", "")[:800]
            source_domain = item.get("source", url)
            if url and content:
                raw_texts.append(content)
                # Index the full page for future retrieval
                try:
                    _web_source.fetch(url)
                except Exception:
                    pass
                snippets.append(f"**{title}** ({source_domain})\n[{url}]\n{content}")

        # Append a sentiment block for the agent to reason over
        if raw_texts:
            snippets.append(
                "\n---\n"
                "**Analyst Sentiment Signals** (extracted from above articles):\n"
                "Look for: upgrades/downgrades, price target changes, recession risk language, "
                "rate cut/hike expectations, earnings revisions. "
                "Classify overall sentiment as: Bullish | Bearish | Mixed — "
                "with a one-sentence rationale citing specific sources above."
            )

        return "\n\n---\n\n".join(snippets) if snippets else "No useful financial news found."

    except ImportError:
        return "tavily-python is not installed. Run: uv add tavily-python"
    except Exception as exc:
        logger.error("search_financial_news failed for '%s': %s", topic, exc)
        return f"Financial news search failed: {exc}"


retrieval_agent = create_agent(
    model=llm,
    tools=[
        search_pgvector,
        get_latest_data,
        fetch_fred_api,
        fetch_web_page,
        search_web_news,
        search_financial_news,
    ],
    system_prompt=RETRIEVAL_AGENT_PROMPT,
)


async def main():
    inputs = {"messages": [HumanMessage(content="What is the most recent Fed Funds rate?")]}
    async for chunk in retrieval_agent.astream(inputs, stream_mode="values"):
        chunk["messages"][-1].pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
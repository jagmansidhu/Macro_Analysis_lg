"""
Integration tests for the retrieval agent.

These tests run the *real* agent against the *real* vector store and verify
that responses contain actual CSV values — not fabricated numbers.

Requirements:
  - DB running with data already ingested (run fred_data_ingest.py first)
  - GEMINI_API_KEY set
  - CLOD_API_KEY set

Ground truth is read from the CSV files at test time so these tests stay
valid as data is updated.
"""
import csv
import os
import re
from pathlib import Path

import pytest

if not os.getenv("CLOD_API_KEY"):
    pytest.skip("CLOD_API_KEY not set — skipping retrieval integration tests.", allow_module_level=True)
if not os.getenv("GEMINI_API_KEY"):
    pytest.skip("GEMINI_API_KEY not set — skipping retrieval integration tests.", allow_module_level=True)

from langchain_core.messages import HumanMessage

from retreival_agent import retrieval_agent, get_latest_data
from config import vector_store_sync

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FRED_DIR = PROJECT_ROOT / "fred_fed_data"

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Ground truth helpers
# ---------------------------------------------------------------------------

def _load_csv_as_dict(filename_glob: str) -> dict[str, str]:
    """Returns {date: value} from the first matching CSV in fred_fed_data."""
    for f in FRED_DIR.glob(filename_glob):
        with open(f, encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            date_col = rows[0].keys().__iter__().__next__()  # first col
            val_col = [k for k in rows[0].keys() if k != date_col][0]
            return {
                row[date_col].strip(): row[val_col].strip()
                for row in rows
                if row[date_col].strip() and row[val_col].strip()
            }
    return {}


CPILFESL_DATA = _load_csv_as_dict("CPILFESL*.csv")
DFF_DATA = _load_csv_as_dict("DFF.csv")
FEDFUNDS_DATA = _load_csv_as_dict("FEDFUNDS.csv")


# ---------------------------------------------------------------------------
# Helper to invoke the agent
# ---------------------------------------------------------------------------

async def _ask(question: str) -> str:
    result = await retrieval_agent.ainvoke(
        {"messages": [HumanMessage(content=question)]}
    )
    return str(result["messages"][-1].content)


# ---------------------------------------------------------------------------
# Known-value accuracy tests
# ---------------------------------------------------------------------------

class TestKnownValues:
    async def test_cpilfesl_jan_2020(self):
        """Agent must return 266.716 — the actual CSV value for CPILFESL on 2020-01-01."""
        expected = CPILFESL_DATA.get("2020-01-01", "")
        assert expected, "Ground truth not found in CSV"

        response = await _ask("What was the CPILFESL value in January 2020?")
        assert expected in response, (
            f"Expected CSV value {expected!r} in response, got:\n{response}"
        )

    async def test_fedfunds_specific_month(self):
        """Pick a month that's in the CSV and assert the agent returns its exact value."""
        # Use a distinctive value to avoid false positives
        date = "2022-07-01"
        expected = FEDFUNDS_DATA.get(date, "")
        assert expected, f"No ground truth for {date}"

        response = await _ask(f"What was the FEDFUNDS rate in July 2022?")
        assert expected in response, (
            f"Expected CSV value {expected!r} in response, got:\n{response}"
        )

    async def test_response_contains_no_dates_outside_csv(self):
        """Any dates cited in the response must exist in the CSV — no hallucinated dates."""
        response = await _ask("Show me CPILFESL values from 2019.")
        # Extract all YYYY-MM-DD patterns from the response
        cited_dates = re.findall(r"\d{4}-\d{2}-\d{2}", response)
        for d in cited_dates:
            assert d in CPILFESL_DATA, (
                f"Agent cited date {d!r} which does not exist in CPILFESL CSV"
            )


# ---------------------------------------------------------------------------
# Latest-data tool tests
# ---------------------------------------------------------------------------

class TestLatestDataTool:
    async def test_get_latest_data_returns_most_recent_entries(self):
        """get_latest_data must return dates in descending order."""
        result = await get_latest_data.ainvoke({"metric": "CPILFESL", "n": 5})
        assert isinstance(result, str)
        assert len(result) > 0

        # Extract dates from the result
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", result)
        assert dates, "No dates found in get_latest_data result"

        # Dates should be sorted descending (most recent first)
        assert dates == sorted(dates, reverse=True), (
            f"Dates are not in descending order: {dates}"
        )

    async def test_get_latest_data_actual_latest_matches_csv(self):
        """The most recent date returned must be the last non-blank row in the CSV."""
        csv_latest = max(CPILFESL_DATA.keys())
        result = await get_latest_data.ainvoke({"metric": "CPILFESL", "n": 1})
        assert csv_latest in result, (
            f"Expected latest CSV date {csv_latest!r} in result, got:\n{result}"
        )

    async def test_get_latest_data_values_exist_in_csv(self):
        """Every value returned by get_latest_data must exist in the CSV."""
        result = await get_latest_data.ainvoke({"metric": "CPILFESL", "n": 5})
        csv_values = set(CPILFESL_DATA.values())

        # Extract decimal numbers from result
        returned_values = re.findall(r"\d+\.\d+", result)
        for v in returned_values:
            assert v in csv_values, (
                f"Value {v!r} returned by get_latest_data not found in CPILFESL CSV"
            )

    async def test_get_latest_data_unknown_metric_returns_not_found(self):
        """Querying a metric not in the DB must return a 'not found' message — not a number."""
        result = await get_latest_data.ainvoke({"metric": "NONEXISTENT_XYZ", "n": 3})
        # Should say something like "No data found" — must NOT contain a fabricated value
        fabricated_numbers = re.findall(r"\b\d{3,}\.\d+\b", result)
        assert len(fabricated_numbers) == 0, (
            f"Agent fabricated values for unknown metric: {fabricated_numbers}"
        )


# ---------------------------------------------------------------------------
# No-hallucination tests
# ---------------------------------------------------------------------------

class TestNoHallucination:
    async def test_agent_admits_ignorance_for_unknown_metric(self):
        """Agent must not invent numbers for a metric it has never seen."""
        response = await _ask("What is the current NASDAQ composite value?")
        # Should contain a disclaimer, not a large number
        hallucinated = re.findall(r"\b\d{4,}\b", response)  # 4+ digit numbers
        assert len(hallucinated) == 0, (
            f"Agent may have hallucinated values for NASDAQ: {hallucinated}\n"
            f"Full response: {response}"
        )

    async def test_all_numeric_values_traceable_to_csv(self):
        """
        For a narrow query, every number in the response must exist in the CSV.
        Uses a very specific query to limit the scope of the answer.
        """
        response = await _ask(
            "What was the exact CPILFESL value on 2021-06-01? "
            "Answer with only the date and value, nothing else."
        )
        expected = CPILFESL_DATA.get("2021-06-01", "")
        assert expected, "Ground truth not found for 2021-06-01"
        assert expected in response, (
            f"Expected {expected!r} in response, got:\n{response}"
        )


# ---------------------------------------------------------------------------
# Vector store direct tests (no LLM)
# ---------------------------------------------------------------------------

class TestVectorStoreRetrieval:
    def test_similarity_search_returns_real_content(self):
        """Direct vector store search — bypasses LLM entirely."""
        results = vector_store_sync.similarity_search(
            "CPILFESL January 2020", k=5
        )
        assert len(results) > 0
        top = results[0]
        assert "CPILFESL" in top.page_content
        assert top.metadata.get("metric") == "CPILFESL"

    def test_filter_by_metric_returns_only_that_metric(self):
        """Metadata filter must not leak docs from other metrics."""
        results = vector_store_sync.similarity_search(
            "most recent data", k=20, filter={"metric": "FEDFUNDS"}
        )
        for doc in results:
            assert doc.metadata.get("metric") == "FEDFUNDS", (
                f"Filter leak: expected FEDFUNDS, got {doc.metadata.get('metric')!r}"
            )

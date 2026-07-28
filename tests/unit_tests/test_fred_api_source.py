"""Unit tests for FredApiSource — FRED REST API ingestion.

All tests mock ``httpx.get`` so no real network calls are made.
"""

import json
from unittest.mock import MagicMock, patch, call

import pytest

from RAG.sources.fred_api_source import FredApiSource, DEFAULT_SERIES


def _make_source(api_key: str = "test-key", observation_limit: int = 10) -> FredApiSource:
    mock_vs = MagicMock()
    mock_rm = MagicMock()
    return FredApiSource(
        api_key=api_key,
        vector_store_sync=mock_vs,
        record_manager=mock_rm,
        observation_limit=observation_limit,
    )


def _fred_response(observations: list[dict]) -> MagicMock:
    """Build a fake httpx response containing the given observations."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"observations": observations}
    return resp


SAMPLE_OBS = [
    {"date": "2025-06-01", "value": "5.33"},
    {"date": "2025-05-01", "value": "5.33"},
    {"date": "2025-04-01", "value": "5.33"},
]


class TestFredApiSourceInit:
    def test_raises_without_api_key(self):
        with pytest.raises(ValueError, match="FRED_API_KEY"):
            FredApiSource(api_key="", vector_store_sync=MagicMock(), record_manager=MagicMock())

    def test_constructs_with_valid_key(self):
        source = _make_source()
        assert source.api_key == "test-key"
        assert source.source_name == "fred_api"


class TestPullSeries:
    def test_happy_path_returns_documents(self):
        source = _make_source()
        with patch("RAG.sources.fred_api_source.httpx.get", return_value=_fred_response(SAMPLE_OBS)):
            docs = source._pull_series("FEDFUNDS")

        assert len(docs) == 3
        assert "FEDFUNDS" in docs[0].page_content
        assert "5.33" in docs[0].page_content
        assert "2025-06-01" in docs[0].page_content

    def test_document_metadata_shape(self):
        source = _make_source()
        with patch("RAG.sources.fred_api_source.httpx.get", return_value=_fred_response(SAMPLE_OBS)):
            docs = source._pull_series("UNRATE")

        doc = docs[0]
        assert doc.metadata["metric"] == "UNRATE"
        assert doc.metadata["file_type"] == "fred_api"
        assert doc.metadata["date"] == "2025-06-01"
        assert doc.metadata["year"] == 2025
        assert doc.metadata["value"] == "5.33"
        assert "fetched_at" in doc.metadata

    def test_source_key_format(self):
        """Source key must be fred_api:{SERIES}:{date} for deduplication."""
        source = _make_source()
        with patch("RAG.sources.fred_api_source.httpx.get", return_value=_fred_response(SAMPLE_OBS)):
            docs = source._pull_series("DFF")

        assert docs[0].metadata["source"] == "fred_api:DFF:2025-06-01"

    def test_filters_missing_value_dot(self):
        """FRED encodes missing observations as '.' — they must be skipped."""
        obs_with_missing = [
            {"date": "2025-06-01", "value": "5.33"},
            {"date": "2025-05-01", "value": "."},   # missing — skip
            {"date": "2025-04-01", "value": ""},    # empty — skip
        ]
        source = _make_source()
        with patch("RAG.sources.fred_api_source.httpx.get", return_value=_fred_response(obs_with_missing)):
            docs = source._pull_series("FEDFUNDS")

        assert len(docs) == 1
        assert docs[0].metadata["date"] == "2025-06-01"

    def test_empty_observations_returns_empty(self):
        source = _make_source()
        with patch("RAG.sources.fred_api_source.httpx.get", return_value=_fred_response([])):
            docs = source._pull_series("FEDFUNDS")

        assert docs == []

    def test_series_id_is_uppercased(self):
        """fetch() normalises lowercase series IDs before calling the API."""
        source = _make_source()
        captured_calls = []

        def fake_get(url, params, timeout):
            captured_calls.append(params["series_id"])
            return _fred_response(SAMPLE_OBS)

        with patch("RAG.sources.fred_api_source.httpx.get", side_effect=fake_get), \
             patch.object(source, "_index_docs", return_value=MagicMock()):
            source.fetch("fedfunds")

        assert captured_calls[0] == "FEDFUNDS"


class TestPullSeriesErrors:
    def test_http_error_returns_empty(self):
        import httpx as _httpx

        source = _make_source()
        error_resp = MagicMock()
        error_resp.text = "Bad Request"
        exc = _httpx.HTTPStatusError("400", request=MagicMock(), response=error_resp)

        with patch("RAG.sources.fred_api_source.httpx.get", side_effect=exc):
            docs = source._pull_series("BADID")

        assert docs == []

    def test_network_error_returns_empty(self):
        import httpx as _httpx

        source = _make_source()
        with patch(
            "RAG.sources.fred_api_source.httpx.get",
            side_effect=_httpx.ConnectError("timeout"),
        ):
            docs = source._pull_series("FEDFUNDS")

        assert docs == []


class TestFetch:
    def test_fetch_calls_pull_and_index(self):
        source = _make_source()
        with patch("RAG.sources.fred_api_source.httpx.get", return_value=_fred_response(SAMPLE_OBS)), \
             patch.object(source, "_index_docs", return_value=MagicMock()) as mock_index:
            docs = source.fetch("FEDFUNDS")

        assert len(docs) == 3
        mock_index.assert_called_once()

    def test_fetch_skips_index_when_no_docs(self):
        source = _make_source()
        with patch("RAG.sources.fred_api_source.httpx.get", return_value=_fred_response([])), \
             patch.object(source, "_index_docs") as mock_index:
            docs = source.fetch("EMPTYSERIES")

        assert docs == []
        mock_index.assert_not_called()


class TestIngestAll:
    def test_ingest_all_refreshes_default_series(self):
        source = _make_source()
        fetched = []

        def fake_pull(series_id):
            fetched.append(series_id)
            return [MagicMock()]  # one doc per series

        with patch.object(source, "_pull_series", side_effect=fake_pull), \
             patch.object(source, "_index_docs", return_value=MagicMock(num_added=1, num_updated=0, num_skipped=0, num_deleted=0, errors=[])):
            result = source.ingest_all()

        assert set(fetched) == set(DEFAULT_SERIES)

    def test_ingest_all_continues_on_error(self):
        """One failing series must not abort the others."""
        source = _make_source()
        call_count = 0

        def flaky_pull(series_id):
            nonlocal call_count
            call_count += 1
            if series_id == DEFAULT_SERIES[0]:
                raise RuntimeError("Network error")
            return [MagicMock()]

        with patch.object(source, "_pull_series", side_effect=flaky_pull), \
             patch.object(source, "_index_docs", return_value=MagicMock(num_added=1, num_updated=0, num_skipped=0, num_deleted=0, errors=[])):
            result = source.ingest_all()

        assert call_count == len(DEFAULT_SERIES)
        assert len(result.errors) == 1

"""Unit tests for fred_data_ingest."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from RAG.fred_data_ingest import (
    run_incremental_ingestion,
    run_full_ingestion,
    load_local_directory,
    _build_local_source,
)


@pytest.fixture
def mock_source():
    with patch("RAG.fred_data_ingest._build_local_source") as mock:
        source = MagicMock()
        mock.return_value = source
        yield source


def test_build_local_source():
    """Test that _build_local_source returns a properly configured LocalFileSource."""
    with patch("RAG.fred_data_ingest.WATCH_DIRS", ["dir1", "dir2"]):
        source = _build_local_source()
        assert len(source.directories) == 2
        assert source.directories[0].name == "dir1"
        assert source.directories[1].name == "dir2"


def test_run_incremental_ingestion(mock_source):
    """Test incremental ingestion calls ingest_all."""
    mock_source.ingest_all.return_value = "Result"
    run_incremental_ingestion()
    mock_source.ingest_all.assert_called_once()


@patch("RAG.sources.local_file_source.load_file")
@patch("RAG.fred_data_ingest.index")
@patch("RAG.fred_data_ingest.mark_file_ingested")
@patch("RAG.fred_data_ingest.time.sleep")
def test_run_full_ingestion(mock_sleep, mock_mark, mock_index, mock_load, mock_source, tmp_path):
    """Test full ingestion flow."""
    # Create a fake directory structure
    d1 = tmp_path / "dir1"
    d1.mkdir()
    f1 = d1 / "file1.csv"
    f1.write_text("data")
    
    mock_source.directories = [d1, tmp_path / "missing"]
    
    # Mock load_file
    fake_doc = MagicMock()
    mock_load.return_value = [fake_doc]
    
    mock_index.return_value = {"num_added": 1}
    
    run_full_ingestion()
    
    # Assertions
    mock_load.assert_called_once_with(f1)
    mock_index.assert_called_once()
    assert mock_index.call_args[1]["docs_source"] == [fake_doc]
    mock_mark.assert_called_once_with(f1)
    mock_sleep.assert_called_once_with(3)


@patch("RAG.sources.local_file_source.load_file")
def test_load_local_directory(mock_load, tmp_path):
    """Test load_local_directory shim."""
    d = tmp_path / "test"
    d.mkdir()
    f1 = d / "data.csv"
    f1.write_text("data")
    
    mock_load.return_value = [MagicMock()]
    
    docs = load_local_directory(str(d), "*.csv")
    
    assert len(docs) == 1
    mock_load.assert_called_once_with(f1)

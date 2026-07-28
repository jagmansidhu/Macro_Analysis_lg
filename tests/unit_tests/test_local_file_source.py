"""Unit tests for LocalFileSource — CSV, PDF, and TXT loading."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from RAG.sources.local_file_source import (
    _load_csv,
    _load_csv_wide,
    _detect_year_columns,
    _load_txt,
    load_file,
)


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


class TestLoadCsv:
    def test_basic_csv(self, tmp_dir):
        f = tmp_dir / "FEDFUNDS.csv"
        f.write_text("DATE,FEDFUNDS\n2025-01-01,5.33\n2025-02-01,5.33\n")
        docs = _load_csv(f)
        assert len(docs) == 2
        assert "FEDFUNDS" in docs[0].page_content
        assert "2025-01-01" in docs[0].page_content
        assert docs[0].metadata["metric"] == "FEDFUNDS"
        assert docs[0].metadata["file_type"] == "csv"

    def test_csv_strips_suffix_number(self, tmp_dir):
        f = tmp_dir / "CPILFESL (1).csv"
        f.write_text("DATE,VALUE\n2025-01-01,315.0\n")
        docs = _load_csv(f)
        assert docs[0].metadata["metric"] == "CPILFESL"

    def test_empty_csv_returns_empty(self, tmp_dir):
        f = tmp_dir / "empty.csv"
        f.write_text("DATE,VALUE\n")
        docs = _load_csv(f)
        assert docs == []

    def test_csv_skips_rows_with_missing_values(self, tmp_dir):
        f = tmp_dir / "gapped.csv"
        f.write_text("DATE,VALUE\n2025-01-01,5.0\n2025-02-01,\n2025-03-01,4.5\n")
        docs = _load_csv(f)
        assert len(docs) == 2

    def test_year_metadata_is_parsed(self, tmp_dir):
        f = tmp_dir / "DFF.csv"
        f.write_text("DATE,DFF\n2023-06-15,5.08\n")
        docs = _load_csv(f)
        assert docs[0].metadata["year"] == 2023


WIDE_CSV = (
    "cntry_code,cntry_name,mv_res_2003,mv_nat_2003,mv_res_2004,mv_nat_2004\r\n"
    "1007,UNITED STATES,0,181900,0,243603\r\n"
    "10189,AUSTRIA,3909,3890,8917,8882\r\n"
)


class TestLoadCsvWide:
    def test_wide_csv_produces_docs(self, tmp_dir):
        f = tmp_dir / "common_stock_data_table.csv"
        f.write_text(WIDE_CSV, encoding="utf-8")
        docs = _load_csv(f)
        # 2 countries × 2 years = 4 documents
        assert len(docs) == 4

    def test_wide_csv_page_content_structure(self, tmp_dir):
        f = tmp_dir / "stock_table.csv"
        f.write_text(WIDE_CSV, encoding="utf-8")
        docs = _load_csv(f)
        us_2003 = next(d for d in docs if "2003" in d.page_content and "UNITED STATES" in d.page_content)
        assert "In 2003" in us_2003.page_content
        assert "mv_res" in us_2003.page_content
        assert "181900" in us_2003.page_content

    def test_wide_csv_year_metadata(self, tmp_dir):
        f = tmp_dir / "wide.csv"
        f.write_text(WIDE_CSV, encoding="utf-8")
        docs = _load_csv(f)
        years = {d.metadata["year"] for d in docs}
        assert years == {2003, 2004}

    def test_wide_csv_date_metadata_is_jan_first(self, tmp_dir):
        f = tmp_dir / "wide2.csv"
        f.write_text(WIDE_CSV, encoding="utf-8")
        docs = _load_csv(f)
        for d in docs:
            assert d.metadata["date"].endswith("-01-01")

    def test_wide_csv_key_cols_in_metadata(self, tmp_dir):
        f = tmp_dir / "wide3.csv"
        f.write_text(WIDE_CSV, encoding="utf-8")
        docs = _load_csv(f)
        for d in docs:
            assert "cntry_name" in d.metadata
            assert "cntry_code" in d.metadata

    def test_detect_year_columns_separates_correctly(self):
        fields = ["cntry_code", "cntry_name", "mv_res_2003", "mv_nat_2003", "mv_res_2004"]
        key_cols, year_map = _detect_year_columns(fields)
        assert key_cols == ["cntry_code", "cntry_name"]
        assert 2003 in year_map and 2004 in year_map
        assert len(year_map[2003]) == 2  # mv_res_2003, mv_nat_2003

    def test_csv_without_date_or_year_cols_returns_empty(self, tmp_dir):
        """Columns with no date pattern AND no year suffix → empty."""
        f = tmp_dir / "nodates.csv"
        f.write_text("NAME,VALUE\nfoo,1.0\n")
        docs = _load_csv(f)
        assert docs == []


class TestLoadTxt:
    def test_basic_txt(self, tmp_dir):
        f = tmp_dir / "notes.txt"
        f.write_text("Fed raised rates by 25bps in March 2025.", encoding="utf-8")
        docs = _load_txt(f)
        assert len(docs) == 1
        assert "Fed raised" in docs[0].page_content
        assert docs[0].metadata["file_type"] == "txt"

    def test_empty_txt_returns_empty(self, tmp_dir):
        f = tmp_dir / "empty.txt"
        f.write_text("   \n", encoding="utf-8")
        docs = _load_txt(f)
        assert docs == []


class TestLoadFileDispatcher:
    def test_dispatches_csv(self, tmp_dir):
        f = tmp_dir / "data.csv"
        f.write_text("DATE,VAL\n2025-01-01,1.0\n")
        docs = load_file(f)
        assert len(docs) == 1

    def test_dispatches_txt(self, tmp_dir):
        f = tmp_dir / "data.txt"
        f.write_text("hello world")
        docs = load_file(f)
        assert len(docs) == 1

    def test_unsupported_extension_returns_empty(self, tmp_dir):
        f = tmp_dir / "data.xlsx"
        f.write_text("not handled")
        docs = load_file(f)
        assert docs == []

    def test_pdf_calls_pypdf(self, tmp_dir):
        """PDF loading should delegate to pypdf (mocked here)."""
        f = tmp_dir / "report.pdf"
        f.write_bytes(b"%PDF-1.4 fake content")

        fake_page = MagicMock()
        fake_page.extract_text.return_value = "Fed meeting minutes summary."
        fake_reader = MagicMock()
        fake_reader.pages = [fake_page]

        # PdfReader is lazily imported inside _load_pdf, so patch at pypdf level
        with patch("pypdf.PdfReader", return_value=fake_reader):
            from RAG.sources.local_file_source import _load_pdf
            docs = _load_pdf(f)

        assert len(docs) == 1
        assert "Fed meeting minutes" in docs[0].page_content
        assert docs[0].metadata["file_type"] == "pdf"
        assert docs[0].metadata["page"] == 1


"""
Unit tests for fred_data_ingest.load_local_directory().

No database, no API calls — pure CSV parsing logic.
All tests use temporary directories with synthetic CSV fixtures.
"""
import csv
import re
from pathlib import Path

from RAG.fred_data_ingest import load_local_directory

def write_csv(tmp_path: Path, filename: str, rows: list[dict], fieldnames: list[str]) -> Path:
    p = tmp_path / filename
    val_col = fieldnames[1] if len(fieldnames) > 1 else "VALUE"
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            remapped = {
                fieldnames[0]: row.get("observation_date", row.get(fieldnames[0], "")),
                val_col: row.get("VALUE", row.get(val_col, "")),
            }
            writer.writerow(remapped)
    return p


STANDARD_ROWS = [
    {"observation_date": "2020-01-01", "VALUE": "1.54"},
    {"observation_date": "2020-02-01", "VALUE": "1.58"},
    {"observation_date": "2020-03-01", "VALUE": "0.65"},
]

class TestPageContentFormat:
    def test_content_matches_expected_template(self, tmp_path):
        write_csv(tmp_path, "FEDFUNDS.csv", STANDARD_ROWS, ["observation_date", "FEDFUNDS"])
        docs = load_local_directory(str(tmp_path), "*.csv")

        assert len(docs) == 3
        for doc in docs:
            assert re.match(
                r"^As of \d{4}-\d{2}-\d{2}, the value for macroeconomic indicator [\w.]+ is .+\.$",
                doc.page_content,
            ), f"Unexpected page_content: {doc.page_content!r}"

    def test_content_contains_actual_csv_value(self, tmp_path):
        write_csv(tmp_path, "FEDFUNDS.csv", STANDARD_ROWS, ["observation_date", "FEDFUNDS"])
        docs = load_local_directory(str(tmp_path), "*.csv")

        values_in_docs = {doc.page_content for doc in docs}
        assert any("1.54" in c for c in values_in_docs)
        assert any("1.58" in c for c in values_in_docs)
        assert any("0.65" in c for c in values_in_docs)

    def test_content_contains_actual_date(self, tmp_path):
        write_csv(tmp_path, "FEDFUNDS.csv", STANDARD_ROWS, ["observation_date", "FEDFUNDS"])
        docs = load_local_directory(str(tmp_path), "*.csv")

        dates = {doc.metadata["date"] for doc in docs}
        assert "2020-01-01" in dates
        assert "2020-02-01" in dates
        assert "2020-03-01" in dates

    def test_no_fabricated_values(self, tmp_path):
        rows = [{"observation_date": "2021-06-01", "DFF": "0.07"}]
        write_csv(tmp_path, "DFF.csv", rows, ["observation_date", "DFF"])
        docs = load_local_directory(str(tmp_path), "*.csv")

        assert len(docs) == 1
        assert "0.07" in docs[0].page_content
        assert "0.08" not in docs[0].page_content
        assert "0.06" not in docs[0].page_content

class TestMetadata:
    def test_all_required_keys_present(self, tmp_path):
        write_csv(tmp_path, "FEDFUNDS.csv", STANDARD_ROWS, ["observation_date", "FEDFUNDS"])
        docs = load_local_directory(str(tmp_path), "*.csv")

        for doc in docs:
            assert "source" in doc.metadata
            assert "metric" in doc.metadata
            assert "date" in doc.metadata
            assert "year" in doc.metadata

    def test_year_extracted_correctly(self, tmp_path):
        write_csv(tmp_path, "FEDFUNDS.csv", STANDARD_ROWS, ["observation_date", "FEDFUNDS"])
        docs = load_local_directory(str(tmp_path), "*.csv")

        years = {doc.metadata["year"] for doc in docs}
        assert years == {2020}

    def test_metric_name_is_stem(self, tmp_path):
        write_csv(tmp_path, "DGS2.csv", STANDARD_ROWS, ["observation_date", "DGS2"])
        docs = load_local_directory(str(tmp_path), "*.csv")

        assert all(doc.metadata["metric"] == "DGS2" for doc in docs)

    def test_metric_name_strips_numbered_suffix(self, tmp_path):
        """CPILFESL (1).csv → metric == 'CPILFESL', not 'CPILFESL (1)'."""
        write_csv(tmp_path, "CPILFESL (1).csv", STANDARD_ROWS, ["observation_date", "CPILFESL"])
        docs = load_local_directory(str(tmp_path), "*.csv")

        assert all(doc.metadata["metric"] == "CPILFESL" for doc in docs)

    def test_source_points_to_file(self, tmp_path):
        p = write_csv(tmp_path, "FEDFUNDS.csv", STANDARD_ROWS, ["observation_date", "FEDFUNDS"])
        docs = load_local_directory(str(tmp_path), "*.csv")

        assert all(doc.metadata["source"] == str(p) for doc in docs)

class TestRowFiltering:
    def test_blank_value_rows_are_skipped(self, tmp_path):
        rows = [
            {"observation_date": "2025-10-01", "CPILFESL": ""},   # blank — skip
            {"observation_date": "2025-11-01", "CPILFESL": "331.043"},
        ]
        write_csv(tmp_path, "CPILFESL.csv", rows, ["observation_date", "CPILFESL"])
        docs = load_local_directory(str(tmp_path), "*.csv")

        assert len(docs) == 1
        assert "331.043" in docs[0].page_content

    def test_blank_date_rows_are_skipped(self, tmp_path):
        p = tmp_path / "FEDFUNDS.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["observation_date", "FEDFUNDS"])
            writer.writeheader()
            writer.writerow({"observation_date": "2020-01-01", "FEDFUNDS": "1.54"})
            writer.writerow({"observation_date": "", "FEDFUNDS": "1.58"})  # blank date
        docs = load_local_directory(str(tmp_path), "*.csv")

        assert len(docs) == 1
        assert "2020-01-01" in docs[0].page_content

    def test_file_with_no_date_column_is_skipped(self, tmp_path):
        rows = [{"name": "foo", "value": "bar"}]
        write_csv(tmp_path, "bad.csv", rows, ["name", "value"])
        docs = load_local_directory(str(tmp_path), "*.csv")

        assert docs == []

    def test_empty_file_produces_no_docs(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("observation_date,FEDFUNDS\n", encoding="utf-8")
        docs = load_local_directory(str(tmp_path), "*.csv")

        assert docs == []

class TestMultiFile:
    def test_multiple_csvs_all_loaded(self, tmp_path):
        write_csv(tmp_path, "DFF.csv", STANDARD_ROWS, ["observation_date", "DFF"])
        write_csv(tmp_path, "FEDFUNDS.csv", STANDARD_ROWS, ["observation_date", "FEDFUNDS"])
        docs = load_local_directory(str(tmp_path), "*.csv")

        metrics = {doc.metadata["metric"] for doc in docs}
        assert "DFF" in metrics
        assert "FEDFUNDS" in metrics
        assert len(docs) == 6  # 3 rows × 2 files

    def test_glob_pattern_respected(self, tmp_path):
        write_csv(tmp_path, "DFF.csv", STANDARD_ROWS, ["observation_date", "DFF"])
        (tmp_path / "notes.txt").write_text("ignored")
        docs = load_local_directory(str(tmp_path), "*.csv")

        assert len(docs) == 3

class TestRealCSVFiles:

    def test_cpilfesl_loads_nonzero_docs(self, fred_data_dir):
        docs = load_local_directory(str(fred_data_dir), "CPILFESL*.csv")
        assert len(docs) > 0

    def test_cpilfesl_jan_2020_value(self, fred_data_dir):
        """266.716 is the CSV ground truth for CPILFESL on 2020-01-01."""
        docs = load_local_directory(str(fred_data_dir), "CPILFESL*.csv")
        jan_2020 = [d for d in docs if d.metadata["date"] == "2020-01-01"]
        assert len(jan_2020) == 1
        assert "266.716" in jan_2020[0].page_content

    def test_no_doc_has_empty_value(self, fred_data_dir):
        docs = load_local_directory(str(fred_data_dir), "*.csv")
        for doc in docs:
            assert not doc.page_content.rstrip(".").endswith("is "), \
                f"Blank value slipped through: {doc.page_content!r}"

    def test_all_docs_have_valid_date_format(self, fred_data_dir):
        date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        docs = load_local_directory(str(fred_data_dir), "*.csv")
        for doc in docs:
            assert date_re.match(doc.metadata["date"]), \
                f"Bad date in metadata: {doc.metadata['date']!r}"

    def test_all_four_metrics_present(self, fred_data_dir):
        docs = load_local_directory(str(fred_data_dir), "*.csv")
        metrics = {doc.metadata["metric"] for doc in docs}
        assert "CPILFESL" in metrics
        assert "DFF" in metrics
        assert "FEDFUNDS" in metrics
        assert "DGS2" in metrics

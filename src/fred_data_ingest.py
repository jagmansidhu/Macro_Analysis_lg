import csv
import re
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

from langchain_core.documents import Document
from langchain_core.indexing import index
from langsmith import traceable

from config import vector_store_sync, record_manager

DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _detect_date_column(fieldnames: list[str], sample_row: dict) -> str | None:
    for col in fieldnames:
        val = (sample_row.get(col) or "").strip()
        if DATE_PATTERN.match(val):
            return col
    return None


def load_local_directory(directory_path: str, glob_pattern: str) -> list[Document]:
    docs = []
    search_path = Path(directory_path)

    for file_path in search_path.glob(glob_pattern):
        try:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                reader.fieldnames = [name.strip() for name in reader.fieldnames if name]

                rows = list(reader)
                if not rows:
                    print(f"Skipping empty file: {file_path}")
                    continue

                date_col = _detect_date_column(reader.fieldnames, rows[0])
                if not date_col:
                    print(f"No date column detected in {file_path} — skipping.")
                    continue

                val_col = [k for k in reader.fieldnames if k != date_col][0]
                metric_name = re.sub(r"\s*\(\d+\)$", "", file_path.stem)

                for row in rows:
                    date = (row.get(date_col) or "").strip()
                    value = (row.get(val_col) or "").strip()

                    if not date or not value:
                        continue

                    text_content = f"As of {date}, the value for macroeconomic indicator {metric_name} is {value}."

                    docs.append(Document(
                        page_content=text_content,
                        metadata={
                            "source": str(file_path),
                            "metric": metric_name,
                            "date": date,
                            "year": int(date.split("-")[0]) if "-" in date else 0
                        }
                    ))
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")

    return docs

@traceable(name="Full Data Ingestion")
def run_ingestion():
    docs = load_local_directory(str(PROJECT_ROOT / 'fred_fed_data'), '*.csv')
    print(f"Loaded {len(docs)} total rows.")

    batch_limit = 500

    for i in range(0, len(docs), batch_limit):
        batch = docs[i:i + batch_limit]
        print(f"Processing batch {i // batch_limit + 1} (Rows {i} to {i + len(batch)})...")

        indexing_result = index(
            docs_source=batch,
            record_manager=record_manager,
            vector_store=vector_store_sync,
            cleanup=None,
            source_id_key="source",
            key_encoder="sha256",
            batch_size=batch_limit
        )

        print(f"Batch Complete: {indexing_result}")

        time.sleep(3)

    print("Full ingestion complete.")


if __name__ == "__main__":
    run_ingestion()
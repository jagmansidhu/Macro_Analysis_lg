import csv
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.indexing import index
from langsmith import traceable

from config import vector_store, record_manager


def load_local_directory(directory_path: str, glob_pattern: str) -> list[Document]:
    docs = []
    search_path = Path(directory_path)

    for file_path in search_path.glob(glob_pattern):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                metric_name = file_path.stem

                for row in reader:
                    date = row.get("DATE") or row.get("date")
                    val_key = [k for k in row.keys() if k.lower() != "date"][0]
                    value = row[val_key]

                    text_content = f"As of {date}, the value for macroeconomic indicator {metric_name} is {value}."

                    docs.append(Document(
                        page_content=text_content,
                        metadata={
                            "source": str(file_path),
                            "metric": metric_name,
                            "date": date,
                            "year": int(date.split("-")[0]) if date else 0
                        }
                    ))
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")

    return docs

@traceable(name="Full Data Ingestion")
def run_ingestion():
    docs = load_local_directory('../fred_fed_data', '*.csv')
    print(f"Loaded {len(docs)} total rows.")

    batch_limit = 500

    for i in range(0, len(docs), batch_limit):
        batch = docs[i:i + batch_limit]
        print(f"Processing batch {i // batch_limit + 1} (Rows {i} to {i + len(batch)})...")

        indexing_result = index(
            docs_source=batch,
            record_manager=record_manager,
            vector_store=vector_store,
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
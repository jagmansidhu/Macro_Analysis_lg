from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langsmith import traceable
from pathlib import Path
from langchain_core.indexing import index

from config import vector_store, record_manager


def load_local_directory(directory_path: str, glob_pattern: str) -> list[Document]:
    docs = []
    search_path = Path(directory_path)

    for file_path in search_path.glob(glob_pattern):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                docs.append(Document(
                    page_content=f.read(),
                    metadata={"source": str(file_path)}
                ))
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")

    return docs

@traceable(name="Full Data Ingestion")
def run_ingestion():
    docs = load_local_directory('../fred_fed_data', '*.csv')
    print(f"Loaded {len(docs)} documents.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(docs)

    indexing_result = index(
        docs_source=all_splits,
        record_manager=record_manager,
        vector_store=vector_store,
        cleanup="incremental",
        source_id_key="source",
        key_encoder = "sha256",
        batch_size=10000
    )

    print(f"Indexing Complete: {indexing_result}")


if __name__ == "__main__":
    run_ingestion()
"""
Integration tests for the FRED data ingestion pipeline.

Requirements:
  - Postgres running and DB_CONNECTION_STRING set (enforced by test_db_smoke.py fixture)
  - GEMINI_API_KEY set (tests skip if not)

Uses a dedicated throw-away collection ('test_collection_ci') — never touches
production data in 'my_docs_v5'.
"""
import os
from pathlib import Path

import psycopg
import pytest
from langchain_core.indexing import index

# Skip entire module if Gemini key is missing
if not os.getenv("GEMINI_API_KEY"):
    pytest.skip("GEMINI_API_KEY not set — skipping ingest integration tests.", allow_module_level=True)

from langchain_classic.indexes import SQLRecordManager
from langchain_postgres import PGVector

from config import embeddings, DB_URL
from RAG.fred_data_ingest import load_local_directory

TEST_COLLECTION = "test_collection_ci"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="session")
def test_vector_store():
    _tmp = PGVector(
        embeddings=embeddings,
        collection_name=TEST_COLLECTION,
        connection=DB_URL,
        async_mode=False,
    )
    _tmp.delete_collection()

    store = PGVector(
        embeddings=embeddings,
        collection_name=TEST_COLLECTION,
        connection=DB_URL,
        async_mode=False,
    )
    yield store
    store.delete_collection()


@pytest.fixture(scope="session")
def test_record_manager(test_vector_store):
    namespace = f"pgvector/{TEST_COLLECTION}"
    rm = SQLRecordManager(namespace=namespace, db_url=DB_URL)
    rm.create_schema()
    existing = rm.list_keys()
    if existing:
        rm.delete_keys(existing)
    yield rm


@pytest.fixture(scope="session")
def cpilfesl_docs(fred_data_dir):
    return load_local_directory(str(fred_data_dir), "CPILFESL*.csv")


@pytest.fixture(scope="session")
def all_docs(fred_data_dir):
    return load_local_directory(str(fred_data_dir), "*.csv")


@pytest.fixture(scope="session")
def ingest_result(test_vector_store, test_record_manager, cpilfesl_docs):
    return index(
        docs_source=cpilfesl_docs,
        record_manager=test_record_manager,
        vector_store=test_vector_store,
        cleanup=None,
        source_id_key="source",
        key_encoder="sha256",
    )


def _count_embeddings_in_db(collection_name: str) -> int:
    dsn = DB_URL.replace("postgresql+psycopg://", "postgresql://")
    with psycopg.connect(dsn) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*)
            FROM langchain_pg_embedding e
                     JOIN langchain_pg_collection c ON c.uuid = e.collection_id
            WHERE c.name = %s
            """,
            (collection_name,),
        ).fetchone()
    return int(row[0]) if row else 0


class TestIngestCorrectness:
    def test_ingest_returns_nonzero_added_count(
            self, ingest_result, cpilfesl_docs
    ):
        assert ingest_result["num_added"] == len(cpilfesl_docs), (
            f"Expected {len(cpilfesl_docs)} docs added, got {ingest_result['num_added']}"
        )
        assert ingest_result.get("num_updated", 0) == 0

    def test_doc_count_matches_csv_row_count(
            self, test_record_manager, cpilfesl_docs
    ):
        indexed_count = len(test_record_manager.list_keys())
        assert indexed_count == len(cpilfesl_docs), (
            f"Record manager has {indexed_count} keys, CSV has {len(cpilfesl_docs)} rows"
        )

    def test_retrieved_content_matches_csv_exactly(self, ingest_result):
        from config import vector_store_sync
        # 2020-01-01 CPILFESL = 266.716 (known ground truth from CSV)
        results = vector_store_sync.similarity_search(
            query="CPILFESL January 2020",
            k=5,
            filter={"metric": "CPILFESL", "date": "2020-01-01"},
        )
        assert len(results) >= 1, "2020-01-01 CPILFESL not found in production store"
        assert "266.716" in results[0].page_content, (
            f"Expected 266.716 in page_content, got: {results[0].page_content!r}"
        )

    def test_metadata_round_trips_correctly(self, ingest_result):
        from config import vector_store_sync
        results = vector_store_sync.similarity_search("CPILFESL 2020", k=10)

        for doc in results:
            assert doc.metadata.get("metric") == "CPILFESL"
            assert "date" in doc.metadata
            assert "year" in doc.metadata
            assert isinstance(doc.metadata["year"], int)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_second_ingest_adds_zero_docs(
            self, test_vector_store, test_record_manager, cpilfesl_docs
    ):
        """Re-indexing the exact same documents must be a complete no-op."""
        result = index(
            docs_source=cpilfesl_docs,
            record_manager=test_record_manager,
            vector_store=test_vector_store,
            cleanup=None,
            source_id_key="source",
            key_encoder="sha256",
        )
        assert result["num_added"] == 0, (
            f"Expected 0 new docs on second ingest, got {result['num_added']}"
        )
        assert result["num_updated"] == 0, (
            f"Expected 0 updated docs on second ingest, got {result['num_updated']}"
        )
        assert result["num_skipped"] == len(cpilfesl_docs), (
            f"Expected all {len(cpilfesl_docs)} docs skipped, got {result['num_skipped']}"
        )

    def test_total_count_unchanged_after_second_ingest(
            self, test_vector_store, test_record_manager, cpilfesl_docs
    ):
        before = _count_embeddings_in_db(TEST_COLLECTION)

        index(
            docs_source=cpilfesl_docs,
            record_manager=test_record_manager,
            vector_store=test_vector_store,
            cleanup=None,
            source_id_key="source",
            key_encoder="sha256",
        )

        after = _count_embeddings_in_db(TEST_COLLECTION)
        assert before == after, f"Embedding count grew from {before} to {after} on duplicate ingest"

    def test_third_ingest_with_one_modified_doc_updates_only_that_doc(
            self, test_vector_store, test_record_manager, cpilfesl_docs
    ):
        """Change one doc's content → only that doc should be updated, others skipped."""
        from langchain_core.documents import Document

        # Mutate a copy of the first doc
        modified = Document(
            page_content=cpilfesl_docs[0].page_content + " [modified]",
            metadata=cpilfesl_docs[0].metadata.copy(),
        )
        modified_batch = [modified] + cpilfesl_docs[1:]

        result = index(
            docs_source=modified_batch,
            record_manager=test_record_manager,
            vector_store=test_vector_store,
            cleanup=None,
            source_id_key="source",
            key_encoder="sha256",
        )
        assert result["num_added"] + result["num_updated"] == 1, (
            f"Expected only 1 doc changed, got added={result['num_added']} "
            f"updated={result['num_updated']}"
        )
        assert result["num_skipped"] == len(cpilfesl_docs) - 1

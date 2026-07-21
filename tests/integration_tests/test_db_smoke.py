import os

import psycopg
import pytest


def _get_db_url() -> str:
    url = os.getenv("DB_CONNECTION_STRING", "")
    return url.replace("postgresql+psycopg://", "postgresql://")

@pytest.fixture(scope="session", autouse=True)
def require_db():
    raw_url = os.getenv("DB_CONNECTION_STRING", "")
    if not raw_url:
        pytest.skip(
            "DB_CONNECTION_STRING not set — skipping all integration tests.",
            allow_module_level=False,
        )

    dsn = raw_url.replace("postgresql+psycopg://", "postgresql://")
    try:
        conn = psycopg.connect(dsn, connect_timeout=5)
        conn.close()
    except Exception as exc:
        pytest.skip(
            f"Postgres not reachable ({exc}) — skipping all integration tests.",
            allow_module_level=False,
        )

class TestDBConnectivity:
    def test_can_connect(self):
        dsn = _get_db_url()
        conn = psycopg.connect(dsn, connect_timeout=5)
        assert conn.closed == 0
        conn.close()

    def test_pgvector_extension_installed(self):
        dsn = _get_db_url()
        with psycopg.connect(dsn) as conn:
            row = conn.execute(
                "SELECT extname FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
        assert row is not None, (
            "pgvector extension not found. Run the app once so langchain-postgres "
            "can CREATE EXTENSION vector, or add it to your init.sql."
        )

    def test_can_execute_vector_cast(self):
        dsn = _get_db_url()
        with psycopg.connect(dsn) as conn:
            result = conn.execute("SELECT '[1,2,3]'::vector <-> '[1,2,3]'::vector").fetchone()
        assert result[0] == pytest.approx(0.0)

    def test_postgres_version_is_14_or_higher(self):
        dsn = _get_db_url()
        with psycopg.connect(dsn) as conn:
            row = conn.execute("SHOW server_version_num").fetchone()
        version_num = int(row[0])
        assert version_num >= 140000, f"Postgres version too old: {row[0]}"

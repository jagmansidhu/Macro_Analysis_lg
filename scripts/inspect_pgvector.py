#!/usr/bin/env python3
"""
inspect_pgvector.py — Show what's currently stored in the PGVector database.

Usage:
    uv run python scripts/inspect_pgvector.py            # summary + web URLs
    uv run python scripts/inspect_pgvector.py --all      # dump every record
    uv run python scripts/inspect_pgvector.py --csv      # export to pgvector_contents.csv
    uv run python scripts/inspect_pgvector.py --type web # filter by file_type
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import COLLECTION_NAME

DB_URL = os.getenv("DB_CONNECTION_STRING", "").replace(
    "postgresql+psycopg://", "postgresql://"
)


def get_conn():
    if not DB_URL:
        print("ERROR: DB_CONNECTION_STRING not set in .env")
        sys.exit(1)
    return psycopg.connect(DB_URL, connect_timeout=5)


def summary(conn):
    """Print a count breakdown by file_type."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                cmetadata->>'file_type' AS file_type,
                COUNT(*)               AS count
            FROM langchain_pg_embedding
            WHERE collection_id = (
                SELECT uuid FROM langchain_pg_collection WHERE name = %s
            )
            GROUP BY file_type
            ORDER BY count DESC
            """,
            (COLLECTION_NAME,),
        )
        rows = cur.fetchall()

    if not rows:
        print(f"Collection '{COLLECTION_NAME}' is empty.")
        return

    total = sum(r[1] for r in rows)
    print(f"\n{'='*50}")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Total documents: {total}")
    print(f"{'='*50}")
    print(f"  {'Type':<15} {'Count':>8}")
    print(f"  {'-'*23}")
    for file_type, count in rows:
        label = file_type or "(no type)"
        print(f"  {label:<15} {count:>8}")
    print()


def web_urls(conn):
    """List all indexed web pages with their URL and fetch timestamp."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                cmetadata->>'url'        AS url,
                cmetadata->>'domain'     AS domain,
                cmetadata->>'fetched_at' AS fetched_at,
                LEFT(document, 120)      AS preview
            FROM langchain_pg_embedding
            WHERE collection_id = (
                SELECT uuid FROM langchain_pg_collection WHERE name = %s
            )
            AND cmetadata->>'file_type' = 'web'
            ORDER BY cmetadata->>'fetched_at' DESC
            """,
            (COLLECTION_NAME,),
        )
        rows = cur.fetchall()

    if not rows:
        print("  No web pages indexed yet.")
        return

    seen_urls = set()
    unique_rows = []
    for url, domain, fetched_at, preview in rows:
        if url not in seen_urls:
            seen_urls.add(url)
            unique_rows.append((url, domain, fetched_at, preview))

    print(f"  Indexed web pages ({len(unique_rows)} unique URLs):")
    print(f"  {'-'*80}")
    for url, domain, fetched_at, preview in unique_rows:
        ts = (fetched_at or "")[:19]
        print(f"\n  [{ts}] {domain}")
        print(f"  {url}")
        print(f"  Preview: {(preview or '').strip()[:100]}...")
    print()


def all_records(conn, file_type_filter: str | None = None):
    """Print all records, optionally filtered by file_type."""
    where_extra = ""
    params: list = [COLLECTION_NAME]
    if file_type_filter:
        where_extra = "AND cmetadata->>'file_type' = %s"
        params.append(file_type_filter)

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT
                cmetadata->>'source'    AS source,
                cmetadata->>'file_type' AS file_type,
                cmetadata->>'metric'    AS metric,
                cmetadata->>'date'      AS date,
                cmetadata->>'url'       AS url,
                LEFT(document, 200)     AS preview
            FROM langchain_pg_embedding
            WHERE collection_id = (
                SELECT uuid FROM langchain_pg_collection WHERE name = %s
            )
            {where_extra}
            ORDER BY cmetadata->>'file_type', cmetadata->>'source', cmetadata->>'date' DESC
            """,
            params,
        )
        rows = cur.fetchall()

    if not rows:
        label = f"type='{file_type_filter}'" if file_type_filter else "all types"
        print(f"  No records found for {label}.")
        return

    for source, file_type, metric, date, url, preview in rows:
        print(f"\n  type={file_type or '?':<10} metric={metric or '-':<15} date={date or '-'}")
        print(f"  source: {source or url or '(unknown)'}")
        print(f"  {(preview or '').strip()[:160]}")
    print()


def export_csv(conn, output_path: Path):
    """Export everything to a CSV for offline browsing."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                cmetadata->>'source'     AS source,
                cmetadata->>'file_type'  AS file_type,
                cmetadata->>'metric'     AS metric,
                cmetadata->>'date'       AS date,
                cmetadata->>'url'        AS url,
                cmetadata->>'domain'     AS domain,
                cmetadata->>'fetched_at' AS fetched_at,
                document
            FROM langchain_pg_embedding
            WHERE collection_id = (
                SELECT uuid FROM langchain_pg_collection WHERE name = %s
            )
            ORDER BY file_type, source
            """,
            (COLLECTION_NAME,),
        )
        rows = cur.fetchall()
        cols = [d.name for d in cur.description]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(rows)

    print(f"  Exported {len(rows)} records -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Inspect PGVector contents")
    parser.add_argument("--all", action="store_true", help="Print all records")
    parser.add_argument("--csv", action="store_true", help="Export to CSV file")
    parser.add_argument(
        "--type",
        dest="file_type",
        help="Filter by file_type: web, csv, fred_api, pdf, txt",
    )
    args = parser.parse_args()

    conn = get_conn()

    summary(conn)

    if args.csv:
        out = PROJECT_ROOT / "pgvector_contents.csv"
        export_csv(conn, out)
    elif args.all or args.file_type:
        all_records(conn, file_type_filter=args.file_type)
    else:
        # Default: show web URLs (the interesting dynamic content)
        print("  -- Web pages indexed by the agent --")
        web_urls(conn)
        print("  Tip: use --all to see every record, --csv to export everything,")
        print("       --type web|csv|fred_api|pdf|txt to filter by source type.\n")

    conn.close()


if __name__ == "__main__":
    main()

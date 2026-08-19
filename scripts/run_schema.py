"""
Runs scripts/schema.sql against Postgres using psycopg2 — an alternative
to the `psql` CLI in case it isn't installed locally.

Usage:
    uv run python scripts/run_schema.py
"""
import os
from pathlib import Path
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_DSN = os.environ.get("PG_DSN", "dbname=logs user=postgres password=postgres host=localhost port=5433")


def main():
    schema_path = Path(__file__).parent / "schema.sql"
    sql = schema_path.read_text()

    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(sql)
    cur.close()
    conn.close()

    print("Schema applied successfully.")


if __name__ == "__main__":
    main()

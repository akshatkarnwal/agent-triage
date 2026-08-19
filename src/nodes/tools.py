"""
Tool nodes.

1. `search_logs_node` — semantic search over log embeddings (pgvector),
   using the same embedding model as scripts/seed_logs.py.
2. `correlate_node` — deterministic SQL time-window correlation by entity
   (IP/user/host). No LLM call — this is the SIEM-domain differentiator,
   the kind of correlation logic DNIF's engine does, reimplemented as an
   agent tool.
"""
import os
import datetime
import psycopg2
import google.generativeai as genai
from src.state import AgentState

DB_DSN = os.environ.get("PG_DSN", "dbname=logs user=postgres password=postgres host=localhost")
EMBED_MODEL = "models/gemini-embedding-001"  # text-embedding-004 was shut down Jan 2026


def _embed(text: str) -> list[float]:
    result = genai.embed_content(model=EMBED_MODEL, content=text, output_dimensionality=768)
    return result["embedding"]


def search_logs_node(state: AgentState) -> dict:
    """Semantic search over log embeddings (pgvector). Uses the same
    embedding model as scripts/seed_logs.py so vectors are comparable."""
    params = state["last_params"]
    query = params.get("query", state["alert_context"])

    query_vec = _embed(query)

    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()
    cur.execute(
        """SELECT log_line, ts, host
           FROM logs
           ORDER BY embedding <-> %s::vector
           LIMIT 10""",
        (query_vec,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    evidence = [f"[{ts}] {host}: {line}" for line, ts, host in rows]

    return {
        "retrieved_evidence": evidence,
        "tool_calls_made": [{
            "tool": "search_logs",
            "params": params,
            "result": f"{len(evidence)} log lines retrieved",
            "iteration": state["iteration_count"],
        }],
    }


def correlate_node(state: AgentState) -> dict:
    """Deterministic time-window correlation for a given entity (IP, user,
    or host). Pure SQL, no LLM call — trustworthy evidence for the report
    vs. an LLM hallucination risk."""
    params = state["last_params"]
    entity = params.get("entity")
    window_min = params.get("time_window_minutes", 30)

    if not entity:
        return {
            "retrieved_evidence": ["correlate called with no entity — skipped"],
            "tool_calls_made": [{
                "tool": "correlate",
                "params": params,
                "result": "no entity provided, skipped",
                "iteration": state["iteration_count"],
            }],
        }

    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()

    # Find the entity's own events first, to anchor the time window around
    # when it actually appeared rather than "now".
    cur.execute(
        """SELECT ts FROM logs
           WHERE source_ip = %s OR user_id = %s OR host = %s
           ORDER BY ts LIMIT 1""",
        (entity, entity, entity),
    )
    anchor_row = cur.fetchone()
    anchor_ts = anchor_row[0] if anchor_row else datetime.datetime.utcnow()

    window_start = anchor_ts - datetime.timedelta(minutes=window_min)
    window_end = anchor_ts + datetime.timedelta(minutes=window_min)

    cur.execute(
        """SELECT log_line, ts, host
           FROM logs
           WHERE (source_ip = %s OR user_id = %s OR host = %s)
             AND ts BETWEEN %s AND %s
           ORDER BY ts""",
        (entity, entity, entity, window_start, window_end),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    correlated = [f"[{ts}] {host}: {line}" for line, ts, host in rows]

    return {
        "retrieved_evidence": correlated,
        "tool_calls_made": [{
            "tool": "correlate",
            "params": params,
            "result": f"{len(correlated)} correlated events for entity={entity} within {window_min}m",
            "iteration": state["iteration_count"],
        }],
    }

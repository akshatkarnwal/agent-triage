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

DB_DSN = os.environ.get(
    "PG_DSN", "dbname=logs user=postgres password=postgres host=localhost port=5433"
)
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

    # Take the 60 nearest rows, then collapse identical or near-identical lines
    # (differing only in long numbers), so 47 copies of one failure, or 60 generated
    # DNS names, can't use up every slot and push rarer, relevant lines out.
    cur.execute(
        """WITH nearest AS (
               SELECT log_line, host, ts, embedding <=> %s::vector AS dist
               FROM logs
               ORDER BY embedding <=> %s::vector
               LIMIT 60
           )
           SELECT min(log_line) AS example, host, count(*) AS n,
                  min(ts) AS first_ts, max(ts) AS last_ts, min(dist) AS dist
           FROM nearest
           GROUP BY regexp_replace(log_line, '[0-9]{4,}', '#', 'g'), host
           ORDER BY min(dist)
           LIMIT 15""",
        (query_vec, query_vec),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    evidence = []
    for line, host, n, first_ts, last_ts, _dist in rows:
        if n == 1:
            evidence.append(f"[{first_ts}] {host}: {line}")
        else:
            evidence.append(f"[{first_ts} to {last_ts}] {host}: {line} (x{n})")

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
    # The planner sometimes prefixes the entity ("host:web-prod-02"); strip it.
    if entity and ":" in entity and entity.split(":", 1)[0] in {"ip", "host", "user"}:
        entity = entity.split(":", 1)[1]
    # Multi-stage activity often spans more than half an hour, so never use a
    # window shorter than 60 minutes whatever the planner asks for.
    window_min = max(int(params.get("time_window_minutes") or 60), 60)

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

    window = datetime.timedelta(minutes=window_min)

    # Anchor the window on this entity's busiest period (the event with the most
    # same-entity events within +/- the window), not on its earliest event ever:
    # a host or user with days of routine history would otherwise be anchored
    # on old noise and miss the incident.
    cur.execute(
        """SELECT a.ts FROM logs a
           WHERE a.source_ip = %s OR a.user_id = %s OR a.host = %s
           ORDER BY (
               SELECT count(*) FROM logs b
               WHERE (b.source_ip = %s OR b.user_id = %s OR b.host = %s)
                 AND b.ts BETWEEN a.ts - %s AND a.ts + %s
           ) DESC, a.ts
           LIMIT 1""",
        (entity,) * 6 + (window, window),
    )
    anchor_row = cur.fetchone()
    anchor_ts = anchor_row[0] if anchor_row else datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)

    window_start = anchor_ts - window
    window_end = anchor_ts + window

    # Group events that differ only in long numbers (e.g. generated DNS names),
    # keeping one example line and a count.
    cur.execute(
        """SELECT min(log_line) AS example, host, count(*) AS n,
                  min(ts) AS first_ts, max(ts) AS last_ts
           FROM logs
           WHERE (source_ip = %s OR user_id = %s OR host = %s)
             AND ts BETWEEN %s AND %s
           GROUP BY regexp_replace(log_line, '[0-9]{4,}', '#', 'g'), host
           ORDER BY min(ts)
           LIMIT 40""",
        (entity, entity, entity, window_start, window_end),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    correlated = []
    for line, host, n, first_ts, last_ts in rows:
        if n == 1:
            correlated.append(f"[{first_ts}] {host}: {line}")
        else:
            correlated.append(f"[{first_ts} to {last_ts}] {host}: {line} (x{n})")

    return {
        "retrieved_evidence": correlated,
        "tool_calls_made": [{
            "tool": "correlate",
            "params": params,
            "result": f"{len(correlated)} correlated events for entity={entity} within {window_min}m",
            "iteration": state["iteration_count"],
        }],
    }
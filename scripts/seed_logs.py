"""
Seeds the `logs` table with synthetic SIEM log data so search_logs_node and
correlate_node have something real to query.

Two kinds of data get generated:
1. "Noise" logs — generic background traffic across a handful of hosts,
   to make retrieval non-trivial (the agent has to actually find the
   relevant needle).
2. Scenario logs — hand-aligned to the three incidents in
   eval/golden_dataset.json, so when you run the RAGAS eval, the agent has
   a real chance of retrieving the right evidence and getting scored
   meaningfully instead of against empty/stub data.

Usage:
    uv run python scripts/seed_logs.py
"""
import os
import time
import random
import datetime
import psycopg2
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

DB_DSN = os.environ.get("PG_DSN", "dbname=logs user=postgres password=postgres host=localhost")
EMBED_MODEL = "models/gemini-embedding-001"  # text-embedding-004 was shut down Jan 2026

HOSTS = ["web-prod-01", "web-prod-02", "db-replica-01", "finance-ws-07", "api-gw-03"]
USERS = ["jdoe", "svc-deploy", "asharma", "mkumar", "svc-backup"]


def embed(text: str, max_retries: int = 5) -> list[float]:
    """Embeds text with a small delay to stay under the free tier's
    100 requests/minute cap, plus retry-with-backoff for the odd 429
    that slips through anyway."""
    for attempt in range(max_retries):
        try:
            result = genai.embed_content(
                model=EMBED_MODEL, content=text, output_dimensionality=768
            )
            time.sleep(0.7)  # ~85 req/min, under the 100/min free-tier cap
            return result["embedding"]
        except ResourceExhausted as e:
            wait = 45 * (attempt + 1)  # back off harder each retry
            print(f"  Rate limited, waiting {wait}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)
    raise RuntimeError(f"embed() failed after {max_retries} retries — quota still exhausted")


def generate_noise_logs(n: int = 200) -> list[dict]:
    """Ask Gemini for generic, benign log lines to act as background noise."""
    prompt = f"""Generate {n} realistic but BENIGN SIEM log lines for a
mid-size company's infrastructure (web servers, a DB replica, workstations,
an API gateway). Mix of auth events, HTTP requests, cron jobs, health
checks, normal user logins. No security incidents, all routine.

Return ONLY a JSON array, no markdown fences, each item shaped as:
{{"log_line": "...", "host": "...", "source_ip": "10.x.x.x or null", "user_id": "... or null"}}
"""
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    raw = response.text.strip().removeprefix("```json").removesuffix("```").strip()

    import json
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print("Warning: noise log generation returned unparseable JSON, skipping batch")
        return []


def scenario_logs() -> list[dict]:
    """Hand-aligned logs for each golden_dataset.json incident, so the
    agent has real evidence to retrieve and correlate against."""
    return [
        # inc-001: SSH brute-force -> success on web-prod-02
        *[{
            "log_line": f"Failed SSH login attempt for user svc-deploy from 203.0.113.45",
            "host": "web-prod-02", "source_ip": "203.0.113.45", "user_id": "svc-deploy",
        } for _ in range(12)],
        {
            "log_line": "Successful SSH login for user svc-deploy from 203.0.113.45",
            "host": "web-prod-02", "source_ip": "203.0.113.45", "user_id": "svc-deploy",
        },
        {
            "log_line": "sshd config check: PasswordAuthentication yes for svc-deploy (policy requires key-only)",
            "host": "web-prod-02", "source_ip": None, "user_id": "svc-deploy",
        },

        # inc-002: DNS tunneling / DGA beaconing on db-replica-01
        *[{
            "log_line": f"Outbound DNS query for {random.randint(1000,9999)}x9k2p.malicious-c2.example from db-replica-01",
            "host": "db-replica-01", "source_ip": None, "user_id": None,
        } for _ in range(15)],
        {
            "log_line": "Firewall log: port 5432 on db-replica-01 reachable from 0.0.0.0/0",
            "host": "db-replica-01", "source_ip": None, "user_id": None,
        },
        {
            "log_line": "Replication service on db-replica-01 running version with known unpatched CVE-2025-XXXX",
            "host": "db-replica-01", "source_ip": None, "user_id": None,
        },

        # inc-003: privilege escalation via jdoe on finance-ws-07
        {
            "log_line": "Phishing email with macro attachment opened by user jdoe",
            "host": "finance-ws-07", "source_ip": None, "user_id": "jdoe",
        },
        {
            "log_line": "PowerShell scheduled task created by jdoe session",
            "host": "finance-ws-07", "source_ip": None, "user_id": "jdoe",
        },
        {
            "log_line": "User jdoe added to local Administrators group via scheduled task",
            "host": "finance-ws-07", "source_ip": None, "user_id": "jdoe",
        },
        {
            "log_line": "Scheduled task deleted immediately after execution in jdoe's PowerShell session",
            "host": "finance-ws-07", "source_ip": None, "user_id": "jdoe",
        },
    ]


def random_ts_within_last_days(days: int = 3) -> datetime.datetime:
    now = datetime.datetime.now(datetime.UTC)
    delta = datetime.timedelta(
        days=random.uniform(0, days),
        seconds=random.uniform(0, 86400),
    )
    return now - delta


def insert_logs(rows: list[dict]):
    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()

    for i, row in enumerate(rows, 1):
        ts = random_ts_within_last_days()
        vec = embed(row["log_line"])
        cur.execute(
            """INSERT INTO logs (log_line, ts, host, source_ip, user_id, embedding)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (row["log_line"], ts, row.get("host"), row.get("source_ip"),
             row.get("user_id"), vec),
        )
        if i % 20 == 0:
            conn.commit()
            print(f"  {i}/{len(rows)} inserted...")

    conn.commit()
    cur.close()
    conn.close()


def main():
    print("Generating scenario logs (aligned to golden_dataset.json)...")
    scenario_rows = scenario_logs()
    print(f"  {len(scenario_rows)} scenario log lines")

    print("Generating noise logs via Gemini...")
    noise_rows = generate_noise_logs(n=150)
    print(f"  {len(noise_rows)} noise log lines")

    all_rows = scenario_rows + noise_rows
    print(f"Embedding + inserting {len(all_rows)} total log lines...")
    insert_logs(all_rows)

    print("Done. Verify with: SELECT count(*) FROM logs;")


if __name__ == "__main__":
    main()
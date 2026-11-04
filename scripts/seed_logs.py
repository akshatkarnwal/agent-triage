"""
Seeds the `logs` table with synthetic SIEM log data so search_logs_node and
correlate_node have something real to query.

Two kinds of data get generated:
1. "Noise" logs — benign background traffic built from templates (no LLM
   call, so no quota use and fully reproducible). It includes look-alike
   events (a mistyped SSH password, a routine group change) so retrieval
   has hard negatives, not just obviously unrelated lines.
2. Scenario logs — aligned to the three incidents in eval/golden_dataset.json
   in both content AND timing (e.g. 47 SSH failures in 6 minutes, then a
   success), so retrieval and time-window correlation have real structure
   to find.

Running it replaces the table's contents, so it is safe to re-run.

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
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

DB_DSN = os.environ.get(
    "PG_DSN", "dbname=logs user=postgres password=postgres host=localhost port=5433"
)
EMBED_MODEL = "models/gemini-embedding-001"  # text-embedding-004 was shut down Jan 2026

HOSTS = ["web-prod-01", "web-prod-02", "db-replica-01", "finance-ws-07", "api-gw-03"]
USERS = ["jdoe", "svc-deploy", "asharma", "mkumar", "svc-backup"]


@lru_cache(maxsize=None)
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


# Hosts by OS, so the noise is internally consistent (no cron jobs or Ubuntu
# update checks on a Windows workstation, no Windows Update on a Linux server).
WINDOWS_HOSTS = ["finance-ws-07"]
LINUX_HOSTS = [h for h in HOSTS if h not in WINDOWS_HOSTS]
HOSTS_BY_OS = {"windows": WINDOWS_HOSTS, "linux": LINUX_HOSTS, "any": HOSTS}

# (template, uses_ip, uses_user, os). The look-alikes read similarly to the
# incidents but are benign, so a retriever has to rank by meaning, not keywords.
BENIGN_TEMPLATES = [
    ("Successful SSH login for user {user} from {ip} (publickey)", True, True, "linux"),
    ("Session closed for user {user}", False, True, "linux"),
    ("GET /health 200 from {ip}", True, False, "linux"),
    ("GET /api/v1/status 200 12ms from {ip}", True, False, "linux"),
    ("cron: nightly backup completed for {host}", False, False, "linux"),
    ("cron: log rotation finished on {host}", False, False, "linux"),
    ("systemd: {host} service check passed", False, False, "linux"),
    ("Outbound DNS query for updates.ubuntu.com from {host}", False, False, "linux"),
    ("User {user} added to group docker by admin (ticket CHG-{n})", False, True, "linux"),
    ("Windows Update installed KB{n} on {host}", False, False, "windows"),
    ("An account was successfully logged on: {user} (interactive)", False, True, "windows"),
    ("Scheduled task 'DiskCleanup' completed successfully on {host}", False, False, "windows"),
    # look-alikes
    ("Failed SSH login attempt for user {user} from {ip}", True, True, "linux"),
    ("Outbound DNS query for api.internal.example from {host}", False, False, "any"),
]


def generate_noise_logs(n: int = 150) -> list[dict]:
    """Benign background traffic from templates: no LLM call, no quota,
    reproducible (fixed seed)."""
    rng = random.Random(42)
    rows = []
    for _ in range(n):
        template, uses_ip, uses_user, os_kind = rng.choice(BENIGN_TEMPLATES)
        host = rng.choice(HOSTS_BY_OS[os_kind])
        ip = f"10.0.{rng.randint(0, 5)}.{rng.randint(2, 250)}" if uses_ip else None
        user = rng.choice(USERS) if uses_user else None
        rows.append({
            "log_line": template.format(host=host, ip=ip, user=user, n=rng.randint(1000, 9999)),
            "host": host,
            "source_ip": ip,
            "user_id": user,
        })
    return rows


def _spread(n, start, seconds):
    """n timestamps evenly spread over `seconds`, beginning at `start`."""
    step = seconds / max(n - 1, 1)
    return [start + datetime.timedelta(seconds=step * i) for i in range(n)]


def scenario_logs() -> list[dict]:
    now = datetime.datetime.now(datetime.UTC)
    td = datetime.timedelta

    # inc-001: 47 failures in 6 minutes, then a success, then the config finding
    t1 = now - td(hours=20)
    fails = _spread(47, t1, 6 * 60)
    ssh = [{
        "log_line": "Failed SSH login attempt for user svc-deploy from 203.0.113.45",
        "host": "web-prod-02", "source_ip": "203.0.113.45", "user_id": "svc-deploy", "ts": ts,
    } for ts in fails]
    ssh += [
        {"log_line": "Successful SSH login for user svc-deploy from 203.0.113.45",
         "host": "web-prod-02", "source_ip": "203.0.113.45", "user_id": "svc-deploy",
         "ts": fails[-1] + td(seconds=30)},
        {"log_line": "sshd config check: PasswordAuthentication yes for svc-deploy (policy requires key-only)",
         "host": "web-prod-02", "source_ip": None, "user_id": "svc-deploy",
         "ts": fails[-1] + td(minutes=3)},
    ]

    # inc-002: 60 DGA-style DNS queries in one minute, firewall and CVE findings just before
    t2 = now - td(hours=10)
    dns = [{
        "log_line": f"Outbound DNS query for {random.randint(1000,9999)}x9k2p.cdn-sync.example from db-replica-01",
        "host": "db-replica-01", "source_ip": None, "user_id": None, "ts": ts,
    } for ts in _spread(60, t2, 60)]
    dns += [
        {"log_line": "Firewall log: port 5432 on db-replica-01 reachable from 0.0.0.0/0",
         "host": "db-replica-01", "source_ip": None, "user_id": None, "ts": t2 - td(minutes=5)},
        {"log_line": "Replication service on db-replica-01 running version with known unpatched CVE-2025-XXXX",
         "host": "db-replica-01", "source_ip": None, "user_id": None, "ts": t2 - td(minutes=5)},
    ]

    # inc-003: phishing, then 38 minutes later the task, the admin add, the cleanup
    t3 = now - td(hours=6)
    priv = [
        {"log_line": "Phishing email with macro attachment opened by user jdoe",
         "host": "finance-ws-07", "source_ip": None, "user_id": "jdoe", "ts": t3},
        {"log_line": "PowerShell scheduled task created by jdoe session",
         "host": "finance-ws-07", "source_ip": None, "user_id": "jdoe", "ts": t3 + td(minutes=38)},
        {"log_line": "User jdoe added to local Administrators group via scheduled task",
         "host": "finance-ws-07", "source_ip": None, "user_id": "jdoe", "ts": t3 + td(minutes=40)},
        {"log_line": "Scheduled task deleted immediately after execution in jdoe's PowerShell session",
         "host": "finance-ws-07", "source_ip": None, "user_id": "jdoe", "ts": t3 + td(minutes=40, seconds=2)},
    ]
    return ssh + dns + priv


def random_ts_within_last_days(days: int = 3) -> datetime.datetime:
    now = datetime.datetime.now(datetime.UTC)
    delta = datetime.timedelta(
        days=random.uniform(0, days),
        seconds=random.uniform(0, 86400),
    )
    return now - delta


def insert_logs(rows: list[dict]):
    # Embed everything BEFORE touching the database, so a quota failure part-way
    # through leaves the existing table untouched.
    prepared = []
    for i, row in enumerate(rows, 1):
        prepared.append((
            row["log_line"],
            row.get("ts") or random_ts_within_last_days(),
            row.get("host"),
            row.get("source_ip"),
            row.get("user_id"),
            embed(row["log_line"]),
        ))
        if i % 20 == 0:
            print(f"  {i}/{len(rows)} embedded...")

    # One transaction: replace the table's contents, so re-running never duplicates.
    conn = psycopg2.connect(DB_DSN)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE logs")
                cur.executemany(
                    """INSERT INTO logs (log_line, ts, host, source_ip, user_id, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    prepared,
                )
    finally:
        conn.close()


def main():
    random.seed(42)  # reproducible DNS subdomains and noise timestamps
    print("Generating scenario logs (aligned to golden_dataset.json)...")
    scenario_rows = scenario_logs()
    print(f"  {len(scenario_rows)} scenario log lines")

    print("Generating noise logs from templates...")
    noise_rows = generate_noise_logs(n=150)
    print(f"  {len(noise_rows)} noise log lines")

    all_rows = scenario_rows + noise_rows
    print(f"Embedding + inserting {len(all_rows)} total log lines...")
    insert_logs(all_rows)

    print("Done. Verify with: SELECT count(*) FROM logs;  (expect "
          f"{len(all_rows)})")


if __name__ == "__main__":
    main()
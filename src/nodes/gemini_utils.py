"""
Shared wrapper around Gemini's generate_content, since the free tier is
only 5 requests/minute AND 20 requests/day for gemini-2.5-flash. Every
node (planner, reflection, report) goes through this instead of calling
the SDK directly, so pacing/backoff/tracking is consistent everywhere.
"""
import time
import json
import datetime
from pathlib import Path
from google.api_core.exceptions import ResourceExhausted

MIN_SECONDS_BETWEEN_CALLS = 13  # 5 req/min = 1 every 12s; add a small buffer
DAILY_LIMIT = 20  # confirmed from this project's actual quota error
_COUNTER_PATH = Path(__file__).parent.parent.parent / ".gemini_call_count.json"


def _today() -> str:
    return datetime.date.today().isoformat()


def _load_counter() -> dict:
    if _COUNTER_PATH.exists():
        data = json.loads(_COUNTER_PATH.read_text())
        if data.get("date") == _today():
            return data
    return {"date": _today(), "count": 0}


def _save_counter(data: dict):
    _COUNTER_PATH.write_text(json.dumps(data))


def get_remaining_quota() -> int:
    """Check how many of today's ~20 free-tier requests are left, without
    making an API call. Call this before starting an expensive run."""
    data = _load_counter()
    return max(0, DAILY_LIMIT - data["count"])


def generate_with_backoff(model, prompt: str, max_retries: int = 5):
    """Calls model.generate_content(prompt), pacing itself to stay under
    the 5 req/min free-tier cap, retrying with growing backoff on any 429
    that slips through, and tracking usage against the known 20/day cap
    so we can fail fast instead of retrying against an exhausted day."""
    data = _load_counter()
    remaining = DAILY_LIMIT - data["count"]
    if remaining <= 0:
        raise RuntimeError(
            f"Local tracker shows today's ~{DAILY_LIMIT} free-tier requests are "
            f"used up (resets after midnight Pacific). Stopping before wasting "
            f"time on retries that will just hit the daily cap again."
        )

    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            data["count"] += 1
            _save_counter(data)
            print(f"  [Gemini call {data['count']}/{DAILY_LIMIT} today]")
            time.sleep(MIN_SECONDS_BETWEEN_CALLS)
            return response
        except ResourceExhausted:
            wait = 40 * (attempt + 1)
            print(f"  Rate limited, waiting {wait}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)
    raise RuntimeError(f"generate_content failed after {max_retries} retries — quota still exhausted")
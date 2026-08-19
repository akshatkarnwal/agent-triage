"""
Checks today's remaining free-tier Gemini quota, tracked locally, without
making any API call. Run this before anything expensive.

Usage:
    uv run python scripts/check_quota.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nodes.gemini_utils import get_remaining_quota, DAILY_LIMIT

remaining = get_remaining_quota()
print(f"Estimated remaining Gemini calls today: {remaining}/{DAILY_LIMIT}")
print("(Only counts calls made through generate_with_backoff — RAGAS's own")
print(" judge calls and seed_logs.py's generation calls aren't tracked here.)")

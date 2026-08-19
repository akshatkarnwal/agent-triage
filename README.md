# Incident Investigation Agent

A multi-step LLM agent that investigates SIEM alerts: given an alert, it
plans an investigation, searches historical logs (pgvector), correlates
related events across a time window, and produces a root-cause report
with a confidence score. Evaluated with RAGAS against a golden dataset.

Built as project 3 of an AI engineering portfolio, following:
- [rag-k8s](https://github.com/akshatkarnwal/rag-k8s) — RAG on Kubernetes
- [runbook-bot](https://github.com/akshatkarnwal/runbook-bot) — Runbook Assistant

## Architecture

```
planner -> [search_logs | correlate] -> reflection -> (loop or) -> report
```

- **Planner** (Gemini): decides the next action as structured JSON
- **search_logs**: pgvector semantic search over log embeddings (reuses
  the rag-k8s retriever)
- **correlate**: deterministic SQL time-window correlation by entity
  (IP/user/host) — no LLM call, the SIEM-domain differentiator
- **reflection** (Gemini): decides whether enough evidence exists to conclude
- **report** (Gemini): writes the final root-cause report

Hard iteration cap (default 4) regardless of what the LLM decides, to
control latency and Gemini free-tier quota.

See `src/graph.py` for the full wiring and routing logic.

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

cp .env.example .env
# then edit .env and paste your real Gemini API key
```

`.env` is loaded automatically by `main.py` and `eval/run_ragas.py` via
`python-dotenv`, and by `docker-compose` when you run `docker-compose up`.
It's already in `.gitignore` — never commit your real key.

### Seed the database

```bash
docker-compose up -d postgres
psql "postgresql://postgres:postgres@localhost:5433/logs" -f scripts/schema.sql
uv run python scripts/seed_logs.py
```

This creates the `logs` table (with a pgvector column) and inserts ~150
synthetic "noise" log lines plus a set of scenario logs hand-aligned to
the three incidents in `eval/golden_dataset.json` — so retrieval and
correlation have real signal to find instead of an empty table.

`src/nodes/tools.py` is fully wired: `search_logs_node` does pgvector
similarity search, `correlate_node` does a time-window SQL join on
whichever entity (IP/user/host) the planner names.

### Run locally

```bash
docker-compose up --build
curl -X POST localhost:8000/investigate \
  -H "Content-Type: application/json" \
  -d '{"alert_context": "Multiple failed SSH attempts from 203.0.113.45"}'
```

### Run the eval suite

```bash
uv run python eval/run_ragas.py --smoke   # 1 case, capped loop — conserves quota
uv run python eval/run_ragas.py           # full 3-case eval, cheap metrics only
uv run python eval/run_ragas.py --full    # adds context_precision/recall (expensive:
                                            # 1 judge-LLM call per retrieved evidence chunk)
```

Scores `faithfulness`, `answer_relevancy` by default (cheap), with
`context_precision`/`context_recall` (retrieval quality) available via
`--full`. Expand `eval/golden_dataset.json` as you find edge cases —
that's the regression suite story for interviews.

Free tier is tight (20 requests/day, 5/min for `gemini-2.5-flash` on
this project) — `src/nodes/gemini_utils.py` paces calls and tracks a
local daily counter (`scripts/check_quota.py`) so you don't burn
retries against an exhausted quota.

### A real finding: faithfulness scores low, and that's expected

First smoke-test run: `faithfulness: 0.29`, `answer_relevancy: 0.84`.

RAGAS's `faithfulness` metric decomposes the report into individual
claims and checks each one against the raw retrieved evidence for
direct support. The evidence here is bare log lines (timestamps, host,
user, IP) — it never states *causes*. So a root-cause narrative and
remediation steps will always score low on this metric, no matter how
sound the reasoning is, because inference isn't literal evidence
restatement.

Rather than chase a misleading high score, the report prompt
(`src/nodes/report.py`) now explicitly separates:
- **Evidence-Based Findings** — only claims traceable to a specific log
  line, which is what `faithfulness` is actually a meaningful measure of
- **Analysis & Recommendations** — the analyst's interpretation and
  next steps, which is expected to include inference RAGAS can't verify

Takeaway for write-ups/interviews: `faithfulness` is the right metric
for the evidence section, not the whole report — pairing it with a
separate reasoning-quality rubric for the analysis section would be the
next step for a fuller eval picture.

## Next steps (not yet built)

- Trace viewer (simple HTML/Streamlit page rendering `tool_calls_made`
  step by step) for demoing the reasoning process
- A separate reasoning-quality rubric for the Analysis & Recommendations
  section, since `faithfulness` only meaningfully covers the Evidence
  section (see finding above)
# Incident Investigation Agent

A multi-step LLM agent that investigates SIEM alerts: given an alert, it
plans an investigation, searches historical logs (pgvector), correlates
related events across a time window, and produces a root-cause report
with a confidence score. Evaluated with RAGAS against a golden dataset.

Built as project 3 of an AI engineering portfolio, following:
- [rag-k8s](https://github.com/akshatkarnwal/rag-k8s): RAG on Kubernetes
- [runbook-bot](https://github.com/akshatkarnwal/runbook-bot): runbook assistant

## Architecture

```
planner -> [search_logs | correlate] -> reflection -> (loop or) -> report
```

- **Planner** (Gemini): decides the next action as structured JSON
- **search_logs**: pgvector semantic search over log embeddings (cosine
  distance, 768-dim Gemini embeddings). Candidate rows are grouped in SQL, first
  by identical line and then by lines that differ only in long numbers, and
  shown with a count, so 47 copies of one event can't fill every result slot
  and push a rarer, more important line out.
- **correlate**: deterministic SQL time-window correlation by entity
  (IP/user/host), no LLM call. The window is centred on the entity's busiest
  period (not its first event ever) and is at least 60 minutes wide, whatever
  the planner asks for. Events are grouped with a count and first/last
  timestamp, e.g. `Failed SSH login ... (x47)`.
- **reflection** (Gemini): decides whether enough evidence exists to conclude
  and reports a confidence value
- **report** (Gemini): writes the final report in three parts: evidence-based
  findings, analysis and recommendations, and a `Verdict` label

Rules enforced in code rather than left to the prompts:
- **Hard iteration cap** (default 4) in the planner and reflection nodes,
  whatever the LLM decides, to control latency and Gemini free-tier quota.
- **A correlation that returned events must exist before the agent may
  conclude** (planner and reflection nodes). A correlate call that finds
  nothing does not count; the planner then retries on the entity named in the
  alert, and never on an entity that already came back empty.

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

This creates the `logs` table (with a pgvector column) and inserts a few
hundred synthetic benign log lines built from templates (consistent with each
host's OS, and including look-alike events such as a single failed login from
an internal IP, with no stated cause) plus scenario logs for the three
incidents in `eval/golden_dataset.json`. The scenario logs have
realistic timing, e.g. 47 SSH failures over 6 minutes followed by a success 30
seconds later, so retrieval and time-window correlation have real structure to
find. Re-running the seed script replaces the table's contents.

### Run locally

```bash
docker-compose up --build
curl -X POST localhost:8000/investigate \
  -H "Content-Type: application/json" \
  -d '{"alert_context": "Multiple failed SSH attempts from 203.0.113.45"}'
```

### Run the eval suite

```bash
uv run python eval/run_ragas.py --smoke          # first case, loop capped at 2 iterations
uv run python eval/run_ragas.py --case inc-002   # one named case
uv run python eval/run_ragas.py                  # all cases
uv run python eval/run_ragas.py --full           # adds context_precision/recall (expensive)
```

Results go to `eval/results[_<id>].json` and the agent's tool calls to
`eval/traces[_<id>].json`.

What is measured:
- **Verdict match:** the report ends with a `Verdict:` label
  (`compromise_confirmed`, `attack_attempt_failed`, `benign_activity` or
  `undetermined`), compared in code against `expected_verdict` in the golden
  dataset. This is the only check of whether the conclusion is right.
- **Faithfulness** (RAGAS), scored on the **Evidence-Based Findings section
  only**, the part required to be traceable to log lines. The Analysis section
  is interpretation, so the verdict check covers it instead.
- **Answer relevancy** (RAGAS), scored on the full report. A weak signal,
  since it penalises long structured answers.

The judge is Gemini 2.5 Flash, the same model family as the agent, so the
RAGAS scores are indicative rather than ground truth. The free tier allows 20
`gemini-2.5-flash` requests per day per project, and the judge's calls count
too, so in practice that is about one case per day.
`src/nodes/gemini_utils.py` paces calls with retry-with-backoff and keeps a
local daily counter (`scripts/check_quota.py`), so a run doesn't burn retries
against an exhausted quota. The counter covers the agent's calls only, not the
judge's.

### Results

Each case was run once before and once after the fixes described below.
"Root cause" is a manual comparison of the report against the reference answer
in the golden dataset, not an automated metric.


|             Case             |   Run  |          Tools used          | Faithfulness |   Relevancy  | Root cause |
|------------------------------|--------|------------------------------|--------------|--------------|------------|
| inc-001 SSH brute force      | before | search, correlate            | 0.86         | 0.74         | match      |
| inc-001 SSH brute force      | after  | search, search, correlate    | 0.78         | not measured | match      |
| inc-002 DNS beaconing        | before | search, correlate (1 event)  | 0.82         | 0.76         | missed     |
| inc-002 DNS beaconing        | after  | search, correlate (3 events) | 0.87         | 0.72         | match      |
| inc-003 privilege escalation | before | search x4, no correlate      | 0.94         | 0.65         | missed     |
| inc-003 privilege escalation | after  | search, correlate (4 events) | 0.86         | 0.73         | match      |


The verdict label matched the expected one in all six runs. "Faithfulness" is
scored on the evidence section only. "Not measured" means the relevancy judge
call timed out. "Root cause" compares the report to the reference answer:
inc-002 before missed the exposed port and the unpatched CVE, and inc-003
before reasoned about the wrong events.

What changed: the agent now correlates in every case and the reports reach the
reference root cause in 3 of 3 cases (1 of 3 before). What did not: the verdict
label was already right in all three before, so it cannot separate good
reasoning from lucky reasoning, and mean faithfulness did not improve
(0.87 before, 0.83 after). With one run per case and a judge that varies, those
differences are noise, not a regression or a gain.

### What the eval found

1. **A single faithfulness score over the whole report is misleading.** It
   fell from 0.97 to 0.23 across runs of the same case as the amount of
   inference in the Analysis section changed, because every inference or
   recommendation counts as an unsupported claim. Faithfulness is now computed
   on the evidence section only, and a code-checked verdict label covers the
   conclusion.
2. **Retrieval hid the key event.** The first realistic run concluded the
   attack had failed. The successful login was never retrieved: 47 identical
   failure lines filled the top-k slots, and benign look-alike logins outranked
   the one real success. Grouping lines with counts, and switching to cosine
   distance (the embeddings are truncated to 768 dims and not unit length, so
   L2 compared magnitude as well as meaning), fixed it.
3. **The agent concluded "human error, mistyped password" on 47 failures in
   6 minutes** and reported confidence 1.0. The seed data's look-alike lines
   stated their own cause, which real logs never do, and the prompts said
   nothing about rates. The seed data no longer includes causes, and the report
   prompt requires weighing a benign and a malicious explanation using volume
   and rate.
4. **A prompt rule did not make the agent correlate.** On inc-003 it ran four
   searches and never correlated, then reasoned from unrelated events. The
   fix was in code: force correlation on the last tool step and block
   concluding without it.
5. **The first version of that guard was satisfied by an empty result.** On
   inc-002 the agent correlated on the domain name, found nothing, and the
   guard counted it as done, so the report missed the exposed port and CVE
   sitting on the host. Now an empty correlation does not count, and the
   planner retries on the alert's entity.
6. **The correlation window itself was wrong.** Anchoring on the entity's
   earliest event ever returned one event for a host with a long history, and a
   30-minute window cut off a phishing email 38 minutes before the alert. The
   window now centres on the entity's busiest period and is at least 60
   minutes.
7. **The agent flags what it can't see.** When a log lacks the authentication
   method, the report lists it as a missing fact instead of inventing it.

## Limitations

- **Confidence is self-reported by the LLM and uncalibrated.** It was 1.0,
  0.95 and 1.0 across the three final runs, each of which lists missing key
  facts.
- **Small, synthetic evaluation.** Three scenarios over generated logs, all
  expecting the same verdict label, so a model that always answered
  `compromise_confirmed` would score 3/3 on the verdict check. There is no
  benign case yet, so false-positive behaviour is untested.
- **No held-out case.** The prompts, the correlation window and the guard were
  all adjusted after seeing failures on these same three scenarios, so the
  "after" results show the fixes work on them, not that they generalise.
- **Single runs.** One run per case per condition; the judge is nondeterministic
  and the same model family as the agent. Answer relevancy timed out on two
  runs.
- **Synthetic-data artifacts.** In the seed data, benign successful SSH logins
  say `(publickey)` and the attacker's success line does not; the agent used
  that difference as evidence of password authentication. A real log source
  may not have it.
- **Numbers can be wrong even when faithful.** `search_logs` groups only the 60
  nearest rows, so a large group can be undercounted (55 of 60 DNS queries),
  and on inc-002 the report added that to the full count from `correlate` and
  stated 115 queries. Faithfulness did not catch it because both lines were in
  the evidence.
- **`correlate` is windowed, so it cannot prove an entity has no history.** The
  planner prompt still says correlation shows whether an entity appears
  earlier in the logs, which is no longer true; the "no prior activity" claim
  on inc-001 happens to be right but is not backed by the tool. The anchor
  query is also quadratic per entity.
- **Untrusted log content reaches the LLM prompts.** In a SIEM an attacker can
  influence log text, so prompt injection through log lines is a real risk
  that is not mitigated here.
- Iteration cap and quota pacing exist because of the free tier; there are no
  automated tests or CI for this project yet.

## Next steps

- A benign golden case (for example, two failed logins then a success from an
  internal IP) and a held-out case, to test false positives and generalisation
- Return first-seen and total-count facts from `correlate`, and fix the planner
  prompt line that claims it shows earlier history
- Group over a larger candidate pool in `search_logs` and deduplicate evidence
  that overlaps between tools, so counts are not double-reported
- Repeat each case several times and report the spread
- A trace viewer rendering `tool_calls_made` step by step
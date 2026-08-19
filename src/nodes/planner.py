"""
Planner node: given the alert + evidence gathered so far, decide the next
action. This is the "multi-step reasoning" core of the agent — force the
LLM into structured JSON output so the graph's conditional edge can route
deterministically instead of parsing free text.
"""
import json
import os
import google.generativeai as genai
from src.state import AgentState
from src.nodes.gemini_utils import generate_with_backoff

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
_model = genai.GenerativeModel("gemini-2.5-flash")

PLANNER_PROMPT = """You are a SIEM triage planner investigating a security alert.

Alert context:
{alert_context}

Evidence gathered so far ({num_evidence} items):
{evidence_summary}

Iteration {iteration} of {max_iterations}.

Decide the single next action. Respond with ONLY valid JSON, no markdown fences:
{{
  "action": "search_logs" | "correlate" | "conclude",
  "reasoning": "one sentence on why",
  "params": {{"query": "...", "entity": "...", "time_window_minutes": 30}}
}}

Rules:
- Use "search_logs" to semantically search log history for related events.
- Use "correlate" once you have a specific entity (IP, user, host) worth
  cross-referencing across a time window.
- Use "conclude" only when you have enough evidence to explain root cause,
  or you have hit the iteration budget.
"""


def planner_node(state: AgentState) -> dict:
    evidence_summary = "\n".join(state["retrieved_evidence"][-5:]) or "(none yet)"

    prompt = PLANNER_PROMPT.format(
        alert_context=state["alert_context"],
        num_evidence=len(state["retrieved_evidence"]),
        evidence_summary=evidence_summary,
        iteration=state["iteration_count"],
        max_iterations=state["max_iterations"],
    )

    response = generate_with_backoff(_model, prompt)
    raw = response.text.strip().removeprefix("```json").removesuffix("```").strip()

    try:
        decision = json.loads(raw)
    except json.JSONDecodeError:
        # Fail safe: force a conclude rather than crashing the graph on a
        # malformed LLM response.
        decision = {
            "action": "conclude",
            "reasoning": "planner returned unparseable output, forcing conclusion",
            "params": {},
        }

    # Hard stop regardless of what the model wants, to control latency/cost
    # on the free tier.
    if state["iteration_count"] >= state["max_iterations"]:
        decision["action"] = "conclude"

    return {
        "plan": decision.get("reasoning", ""),
        "next_action": decision["action"],
        "messages": [{"role": "assistant", "content": json.dumps(decision)}],
        "iteration_count": state["iteration_count"] + 1,
        "last_params": decision.get("params", {}),
    }
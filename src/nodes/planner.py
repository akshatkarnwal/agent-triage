"""
Planner node: given the alert + evidence gathered so far, decide the next
action. This is the "multi-step reasoning" core of the agent — force the
LLM into structured JSON output so the graph's conditional edge can route
deterministically instead of parsing free text.
"""
import json
import os
import re
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
Tools already used: {tools_used}

Decide the single next action. Respond with ONLY valid JSON, no markdown fences:
{{
  "action": "search_logs" | "correlate" | "conclude",
  "reasoning": "one sentence on why",
  "params": {{"query": "...", "entity": "...", "time_window_minutes": 60}}
}}

Rules:
- Use "search_logs" to semantically search log history for related events.
- Use "correlate" to pull every event for one specific entity across a time
  window. The entity must be a bare value (for example 203.0.113.45,
  svc-deploy or web-prod-02) with NO prefix such as "ip:" or "host:".
- Normally search first, then correlate the most suspicious entity (usually
  the unfamiliar source IP named in the alert). Correlation also shows whether
  that entity appears earlier in the logs, which matters for judging an alert.
- Use "conclude" only after "correlate" has run, or when you have hit the
  iteration budget.
"""

_IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_USER_RE = re.compile(r"\buser\s+([A-Za-z0-9_.-]+)", re.IGNORECASE)
_HOST_RE = re.compile(r"\b[a-z][a-z0-9]*(?:-[a-z0-9]+)+\b")


def _entity_from_alert(alert: str) -> str | None:
    """Fallback entity when the model asks to correlate without naming one:
    first IP in the alert, else a named user, else a hostname."""
    for pattern in (_IP_RE, _USER_RE, _HOST_RE):
        match = pattern.search(alert)
        if match:
            return match.group(1) if pattern is _USER_RE else match.group(0)
    return None


def _tools_used(state: AgentState) -> str:
    calls = state.get("tool_calls_made", [])
    if not calls:
        return "(none yet)"
    return "; ".join(
        f'{c["tool"]}({c["params"].get("entity") or c["params"].get("query", "")})'
        for c in calls
    )


def _empty_correlate_entities(state: AgentState) -> set:
    """Entities whose correlate call ran but found nothing."""
    return {
        c["params"].get("entity")
        for c in state.get("tool_calls_made", [])
        if c["tool"] == "correlate" and c["result"].startswith("0 correlated")
    }


def _has_correlated(state: AgentState) -> bool:
    """True once a correlate call has run and found events (not skipped, not empty)."""
    return any(
        c["tool"] == "correlate"
        and "skipped" not in c["result"]
        and not c["result"].startswith("0 correlated")
        for c in state.get("tool_calls_made", [])
    )


def planner_node(state: AgentState) -> dict:
    evidence_summary = "\n".join(state["retrieved_evidence"][-5:]) or "(none yet)"

    prompt = PLANNER_PROMPT.format(
        alert_context=state["alert_context"],
        num_evidence=len(state["retrieved_evidence"]),
        evidence_summary=evidence_summary,
        iteration=state["iteration_count"],
        max_iterations=state["max_iterations"],
        tools_used=_tools_used(state),
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

    # Deterministic rule: SQL correlation must run before the agent can finish.
    # Force it when the model tries to conclude early, and also on the LAST tool
    # step, so a run of searches can't use up the whole iteration budget.
    last_tool_step = state["iteration_count"] >= state["max_iterations"] - 1
    if (
        decision.get("action") != "correlate"
        and not _has_correlated(state)
        and state["iteration_count"] < state["max_iterations"]
        and (decision.get("action") == "conclude" or last_tool_step)
    ):
        decision["action"] = "correlate"
        decision["reasoning"] = "correlation has not run yet; required before concluding"

    # Correlate on the alert's own entity when none was named, or when the
    # chosen entity already came back empty.
    if decision.get("action") == "correlate":
        params = decision.setdefault("params", {})
        empty = _empty_correlate_entities(state)
        if not params.get("entity") or params["entity"] in empty:
            entity = _entity_from_alert(state["alert_context"])
            if entity and entity not in empty:
                params["entity"] = entity

    # Hard stop regardless of what the model wants, to control latency/cost
    # on the free tier. This runs last, so the iteration cap always wins.
    if state["iteration_count"] >= state["max_iterations"]:
        decision["action"] = "conclude"

    return {
        "plan": decision.get("reasoning", ""),
        "next_action": decision["action"],
        "messages": [{"role": "assistant", "content": json.dumps(decision)}],
        "iteration_count": state["iteration_count"] + 1,
        "last_params": decision.get("params", {}),
    }
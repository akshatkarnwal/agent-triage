"""
Reflection node: after each tool call, ask "is there enough evidence to
conclude, or do we loop back to the planner?" Separated from the planner
so you can tune/eval it independently (RAGAS can score this step's
decision quality against your golden dataset).
"""
import json
import os
import google.generativeai as genai
from src.state import AgentState
from src.nodes.gemini_utils import generate_with_backoff

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
_model = genai.GenerativeModel("gemini-2.5-flash")

REFLECTION_PROMPT = """You are reviewing evidence gathered during a SIEM alert investigation.

Alert: {alert_context}

Evidence gathered so far:
{evidence}

Respond with ONLY valid JSON, no markdown fences:
{{"sufficient": true|false, "confidence": 0.0-1.0, "reasoning": "one sentence"}}
"""


def reflection_node(state: AgentState) -> dict:
    evidence = "\n".join(state["retrieved_evidence"]) or "(none yet)"
    prompt = REFLECTION_PROMPT.format(
        alert_context=state["alert_context"], evidence=evidence
    )

    response = generate_with_backoff(_model, prompt)
    raw = response.text.strip().removeprefix("```json").removesuffix("```").strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"sufficient": True, "confidence": 0.3, "reasoning": "parse failure, forcing stop"}

    # Route back to planner unless sufficient or out of budget
    should_conclude = result["sufficient"] or state["iteration_count"] >= state["max_iterations"]

    return {
        "confidence_score": result["confidence"],
        "next_action": "conclude" if should_conclude else "continue",
    }

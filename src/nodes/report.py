"""
Report writer node: final synthesis call. Kept separate from the
planner/reflection prompts so RAGAS can score faithfulness/relevancy on
this output specifically against the evidence context.
"""
import os
import google.generativeai as genai
from src.state import AgentState
from src.nodes.gemini_utils import generate_with_backoff

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
_model = genai.GenerativeModel("gemini-2.5-flash")

REPORT_PROMPT = """You are writing a root-cause report for a SIEM alert investigation.

Alert: {alert_context}

Evidence gathered ({num_iterations} investigation steps):
{evidence}

Confidence score: {confidence}

Write the report in two clearly separated parts:

## Evidence-Based Findings
State ONLY facts directly present in the evidence above (timestamps, hosts,
IPs, users, event counts/patterns). Every sentence here should be traceable
to a specific log line. No speculation, no inferred causes, no
recommendations — just what the evidence literally shows.

## Analysis & Recommendations
This is where you interpret the findings: propose a root cause hypothesis,
explain your reasoning, and give remediation steps. Be clear this section
is analyst interpretation built on the evidence above, not a direct
restatement of it — reasonable inference is expected and valuable here.

## Confidence: {confidence}
"""


def report_node(state: AgentState) -> dict:
    evidence = "\n".join(state["retrieved_evidence"]) or "(no evidence gathered)"
    prompt = REPORT_PROMPT.format(
        alert_context=state["alert_context"],
        num_iterations=state["iteration_count"],
        evidence=evidence,
        confidence=state.get("confidence_score", 0.0),
    )

    response = generate_with_backoff(_model, prompt)

    return {"final_report": response.text}
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

Evidence gathered ({num_iterations} investigation steps). A suffix "N identical or near-identical events (differing only in long numbers)" means
N identical events occurred between the two timestamps shown:
{evidence}

Self-reported confidence from the investigation step: {confidence}

Write the report in these parts, using these exact headings:

## Evidence-Based Findings
State ONLY facts directly present in the evidence above (timestamps, hosts,
IPs, users, event counts/patterns). Every sentence here should be traceable
to a specific log line. No speculation, no inferred causes, no
recommendations; just what the evidence literally shows.

## Analysis & Recommendations
This is where you interpret the findings: propose a root cause hypothesis,
explain your reasoning, and give remediation steps. This section is analyst
interpretation built on the evidence above, not a restatement of it.
Your reasoning must:
- weigh at least one benign explanation and one malicious explanation, and say
  which the evidence favours and why;
- use volume and rate (a handful of failures over minutes is typical of a
  person; dozens within minutes is automated);
- note whether the source is internal (10.x, 172.16-31.x, 192.168.x) or
  external, and whether the evidence shows earlier history for it;
- say plainly when a key fact is missing, instead of assuming it.

## Verdict: <exactly one label>
Choose the single label that best fits, and put it on this heading line:
- compromise_confirmed: malicious activity that succeeded
- attack_attempt_failed: malicious activity with no evidence it succeeded
- benign_activity: a routine or benign explanation fits better
- undetermined: the evidence is not sufficient to decide

## Confidence: {confidence}
"""


def report_node(state: AgentState) -> dict:
    evidence = "\n".join(dict.fromkeys(state["retrieved_evidence"])) or "(no evidence gathered)"
    prompt = REPORT_PROMPT.format(
        alert_context=state["alert_context"],
        num_iterations=state["iteration_count"],
        evidence=evidence,
        confidence=state.get("confidence_score", 0.0),
    )

    response = generate_with_backoff(_model, prompt)

    return {"final_report": response.text}
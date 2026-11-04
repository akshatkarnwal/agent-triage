"""
Regression-style eval for the investigation agent. Run it after any prompt or
graph change to catch quality drift.

What it measures
  - verdict match:    the report's `## Verdict:` label vs `expected_verdict` in
                      the golden dataset. Computed in code, no judge calls, and
                      it is the only check of whether the conclusion is RIGHT.
  - faithfulness:     scored on the Evidence-Based Findings section ONLY, since
                      that is the part required to be traceable to log lines.
                      The Analysis section is interpretation and is judged by
                      the verdict check instead.
  - answer_relevancy: scored on the full report.
Faithfulness and relevancy are judged by Gemini, the same model family as the
agent, so treat them as indicative, not as ground truth.

Usage:
    uv run python eval/run_ragas.py            # all golden cases, 2 cheap metrics
    uv run python eval/run_ragas.py --smoke    # 1 case, capped loop, conserves quota
    uv run python eval/run_ragas.py --case inc-002   # one named case; saves
                                                 # results_inc-002.json / traces_inc-002.json
    uv run python eval/run_ragas.py --full     # adds context_precision/recall
                                                 # (expensive: 1 call per evidence chunk)

Outputs: eval/results.json (scores + report) and eval/traces.json (tool calls).
"""
import os
import re
import sys
import types
import json
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()  # must run before importing src.graph, same reason as main.py

# ragas unconditionally imports ChatVertexAI from langchain_community, even
# though we never use Vertex AI, and langchain_community has since removed
# that submodule. Stub it out so ragas's dead import path doesn't crash us.
_vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")
setattr(_vertexai_stub, "ChatVertexAI", type("ChatVertexAI", (), {}))
sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub

from datasets import Dataset
from pydantic import SecretStr
from ragas import evaluate
from ragas.dataset_schema import EvaluationResult
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src.graph import run_investigation

# RAGAS defaults to OpenAI for its judge LLM and embeddings. We only have a
# Gemini key, so wire it to use that instead.
_judge_llm = LangchainLLMWrapper(
    ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.environ["GEMINI_API_KEY"])
)
_judge_embeddings = LangchainEmbeddingsWrapper(
    GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        api_key=SecretStr(os.environ["GEMINI_API_KEY"]),
    )
)

EVIDENCE_HEADING = "## Evidence-Based Findings"
ANALYSIS_HEADING = "## Analysis & Recommendations"
VERDICT_RE = re.compile(r"^#*\s*Verdict:\s*`?([A-Za-z_]+)`?", re.IGNORECASE | re.MULTILINE)


def evidence_section(report: str) -> str:
    """The part of the report that must be traceable to log lines."""
    report = report or ""
    head = report.split(ANALYSIS_HEADING)[0]
    text = head.replace(EVIDENCE_HEADING, "").strip()
    return text or report


def extract_verdict(report: str) -> str:
    match = VERDICT_RE.search(report or "")
    return match.group(1).lower() if match else "missing"


def load_golden_dataset(smoke: bool = False, case_id: str | None = None) -> list[dict]:
    path = Path(__file__).parent / "golden_dataset.json"
    cases = json.loads(path.read_text())
    if case_id:
        cases = [c for c in cases if c["id"] == case_id]
        if not cases:
            raise SystemExit(f"No case with id {case_id!r} in golden_dataset.json")
        print(f"Running only case {case_id}")
    elif smoke:
        cases = cases[:1]
        print(f"Smoke test mode: running only the first case ({cases[0]['id']})")
    return cases


def run_agent_on_dataset(cases: list[dict], max_iterations: int = 4,
                         traces_name: str = "traces.json") -> tuple[dict, dict]:
    """Run the full agent on each case. Returns the RAGAS columns plus extras
    (evidence-only answers, verdicts) kept outside the RAGAS dataset."""
    questions, answers, contexts, ground_truths = [], [], [], []
    evidence_only, verdicts, expected = [], [], []
    traces = []

    for case in cases:
        print(f"Running {case['id']}...")
        final_state = run_investigation(case["alert_context"], max_iterations=max_iterations)
        report = final_state["final_report"] or ""

        traces.append({
            "id": case["id"],
            "iterations": final_state["iteration_count"],
            "tool_calls": final_state["tool_calls_made"],
            "confidence": final_state.get("confidence_score"),
        })
        questions.append(case["alert_context"])
        answers.append(report)
        contexts.append(list(dict.fromkeys(final_state["retrieved_evidence"])) or [""])
        ground_truths.append(case["ground_truth_root_cause"])
        evidence_only.append(evidence_section(report))
        verdicts.append(extract_verdict(report))
        expected.append(case.get("expected_verdict"))

    (Path(__file__).parent / traces_name).write_text(json.dumps(traces, indent=2, default=str))

    eval_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    extras = {"evidence_only": evidence_only, "verdicts": verdicts, "expected": expected}
    return eval_data, extras


def main():
    smoke = "--smoke" in sys.argv
    full = "--full" in sys.argv
    case_id = sys.argv[sys.argv.index("--case") + 1] if "--case" in sys.argv else None
    suffix = f"_{case_id}" if case_id else ""
    cases = load_golden_dataset(smoke=smoke, case_id=case_id)
    # Smoke mode also caps the agent's own loop: max_iterations=4 can cost up
    # to 9 calls (planner + reflection per loop, plus report) for ONE case.
    max_iterations = 2 if smoke else 4
    eval_data, extras = run_agent_on_dataset(
        cases, max_iterations=max_iterations, traces_name=f"traces{suffix}.json"
    )

    # Free tier is only 5 req/min: run judge calls one at a time.
    run_config = RunConfig(max_workers=1, timeout=300)

    # Faithfulness (and the retrieval metrics) on the evidence section only.
    evidence_data = {**eval_data, "answer": extras["evidence_only"]}
    grounding_metrics = [faithfulness] + ([context_precision, context_recall] if full else [])
    grounding = cast(EvaluationResult, evaluate(
        Dataset.from_dict(evidence_data),
        metrics=grounding_metrics,
        llm=_judge_llm,
        embeddings=_judge_embeddings,
        run_config=run_config,
    ))

    # Relevancy on the full report.
    relevancy = cast(EvaluationResult, evaluate(
        Dataset.from_dict(eval_data),
        metrics=[answer_relevancy],
        llm=_judge_llm,
        embeddings=_judge_embeddings,
        run_config=run_config,
    ))

    out = grounding.to_pandas()
    out["evidence_section"] = out["response"]
    out["response"] = eval_data["answer"]
    out["answer_relevancy"] = relevancy.to_pandas()["answer_relevancy"].values
    out["verdict"] = extras["verdicts"]
    out["expected_verdict"] = extras["expected"]
    out["verdict_match"] = [v == e for v, e in zip(extras["verdicts"], extras["expected"])]

    matches = sum(out["verdict_match"])
    print("\n=== Eval results ===")
    print(f"cases:                       {len(out)}")
    print(f"verdict match:               {matches}/{len(out)}  {list(zip(extras['verdicts'], extras['expected']))}")
    print(f"faithfulness (evidence):     {out['faithfulness'].mean():.4f}")
    print(f"answer_relevancy (report):   {out['answer_relevancy'].mean():.4f}")

    out_path = Path(__file__).parent / f"results{suffix}.json"
    out.to_json(out_path, orient="records", indent=2)
    print(f"\nSaved detailed results to {out_path}")


if __name__ == "__main__":
    main()
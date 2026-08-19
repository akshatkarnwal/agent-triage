"""
Regression-style eval for the agent. Run this after any prompt/graph
change to catch quality drift — this is the artifact that actually
differentiates the project in an interview: a quantified eval suite,
not just "it works when I tried it".

Usage:
    uv run python eval/run_ragas.py            # full 3-case eval, 2 cheap metrics
    uv run python eval/run_ragas.py --smoke    # 1 case, capped loop, conserves quota
    uv run python eval/run_ragas.py --full     # include context_precision/recall too
                                                 # (expensive — 1 call per evidence chunk)

Requires: pip install ragas datasets langchain-google-genai
"""
import os
import sys
import types
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()  # must run before importing src.graph, same reason as main.py

# ragas unconditionally imports ChatVertexAI from langchain_community, even
# though we never use Vertex AI — and langchain_community has since removed
# that submodule as part of its sunset. Stub it out so ragas's dead import
# path doesn't crash us; this is dead code we never call.
_vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_vertexai_stub.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub

from datasets import Dataset
from ragas import evaluate
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

# RAGAS defaults to OpenAI for its judge LLM and embeddings — we only have
# a Gemini key, so wire it to use that instead.
_judge_llm = LangchainLLMWrapper(
    ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.environ["GEMINI_API_KEY"])
)
_judge_embeddings = LangchainEmbeddingsWrapper(
    GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", google_api_key=os.environ["GEMINI_API_KEY"]
    )
)


def load_golden_dataset(smoke: bool = False) -> list[dict]:
    path = Path(__file__).parent / "golden_dataset.json"
    cases = json.loads(path.read_text())
    if smoke:
        cases = cases[:1]
        print(f"Smoke test mode: running only the first case ({cases[0]['id']})")
    return cases


def run_agent_on_dataset(cases: list[dict], max_iterations: int = 4) -> dict:
    """Run the full agent on each case and shape output into the
    {question, answer, contexts, ground_truth} format RAGAS expects."""
    questions, answers, contexts, ground_truths = [], [], [], []

    for case in cases:
        print(f"Running {case['id']}...")
        final_state = run_investigation(case["alert_context"], max_iterations=max_iterations)

        questions.append(case["alert_context"])
        answers.append(final_state["final_report"] or "")
        contexts.append(final_state["retrieved_evidence"] or [""])
        ground_truths.append(case["ground_truth_root_cause"])

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }


def main():
    smoke = "--smoke" in sys.argv
    cases = load_golden_dataset(smoke=smoke)
    # Smoke mode also caps the agent's own loop — max_iterations=4 can cost
    # up to 9 calls (planner+reflection per loop, plus report) for ONE
    # investigation, before RAGAS even starts.
    max_iterations = 2 if smoke else 4
    eval_data = run_agent_on_dataset(cases, max_iterations=max_iterations)
    dataset = Dataset.from_dict(eval_data)

    # context_precision/context_recall each make one judge-LLM call PER
    # retrieved evidence chunk — with ~10 chunks per case, either metric
    # alone can exceed a 20 req/day free-tier budget. Default to the two
    # cheap metrics; pass --full to include all four when quota allows.
    full = "--full" in sys.argv
    metrics = (
        [context_precision, context_recall, faithfulness, answer_relevancy]
        if full else
        [faithfulness, answer_relevancy]
    )
    if not full:
        print("Running cheap metrics only (faithfulness, answer_relevancy). Use --full for all four.")

    # Free tier is only 5 req/min — force RAGAS to run judge calls one at a
    # time (default concurrency fires many at once and most time out).
    run_config = RunConfig(max_workers=1, timeout=300)

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=_judge_llm,
        embeddings=_judge_embeddings,
        run_config=run_config,
    )

    print("\n=== RAGAS Eval Results ===")
    print(result)

    # Save for tracking across runs (build a trend line once you have a
    # few of these — that's the interview-ready artifact)
    out_path = Path(__file__).parent / "results.json"
    result_df = result.to_pandas()
    result_df.to_json(out_path, orient="records", indent=2)
    print(f"\nSaved detailed results to {out_path}")


if __name__ == "__main__":
    main()
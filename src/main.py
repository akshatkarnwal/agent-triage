"""
FastAPI wrapper around the LangGraph agent.

POST /investigate  {"alert_context": "..."} -> full final state (report + trace)
GET  /health
"""
from dotenv import load_dotenv
load_dotenv()  # must run before src.graph imports the node modules, since
                # planner.py/reflection.py/report.py call genai.configure()
                # at import time using os.environ["GEMINI_API_KEY"]

from fastapi import FastAPI
from pydantic import BaseModel
from src.graph import run_investigation

app = FastAPI(title="Incident Investigation Agent")


class InvestigateRequest(BaseModel):
    alert_context: str
    max_iterations: int = 4


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/investigate")
def investigate(req: InvestigateRequest):
    final_state = run_investigation(req.alert_context, req.max_iterations)
    return {
        "report": final_state["final_report"],
        "confidence": final_state["confidence_score"],
        "iterations": final_state["iteration_count"],
        "trace": final_state["tool_calls_made"],  # for the demo trace-viewer
    }

"""
Shared state object passed between all nodes in the LangGraph.

Keep this flat and JSON-serializable — it gets logged at every step for
the RAGAS eval harness and the trace viewer.
"""
from typing import TypedDict, List, Dict, Any, Optional
from typing_extensions import Annotated
import operator


class ToolCallRecord(TypedDict):
    tool: str
    params: Dict[str, Any]
    result: str
    iteration: int


class AgentState(TypedDict):
    # --- input ---
    alert_context: str            # raw alert text / metadata fed in at start
    messages: Annotated[List[Dict[str, str]], operator.add]  # chat history for LLM calls

    # --- planning ---
    plan: Optional[str]           # current step's reasoning/action from planner
    next_action: Optional[str]    # one of: "search_logs" | "correlate" | "conclude"
    last_params: Dict[str, Any]   # params the planner attached to next_action

    # --- evidence gathered so far ---
    tool_calls_made: Annotated[List[ToolCallRecord], operator.add]
    retrieved_evidence: Annotated[List[str], operator.add]

    # --- control flow ---
    iteration_count: int
    max_iterations: int

    # --- output ---
    confidence_score: Optional[float]
    final_report: Optional[str]

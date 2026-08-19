"""
LangGraph wiring. Flow:

    planner -> (route on next_action) -> search_logs | correlate -> reflection
                                                                        │
                                                        ┌───────────────┴───┐
                                                        ▼                   ▼
                                                    planner (loop)      report -> END

The planner picks the tool; reflection decides whether to loop back to the
planner or move to the report. iteration_count caps the loop regardless of
what the LLM decides, so a flaky prompt can't burn your Gemini free-tier
quota in one run.
"""
from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.nodes.planner import planner_node
from src.nodes.tools import search_logs_node, correlate_node
from src.nodes.reflection import reflection_node
from src.nodes.report import report_node


def route_after_planner(state: AgentState) -> str:
    action = state["next_action"]
    if action == "search_logs":
        return "search_logs"
    if action == "correlate":
        return "correlate"
    return "report"  # "conclude" or anything unexpected -> go straight to report


def route_after_reflection(state: AgentState) -> str:
    return "report" if state["next_action"] == "conclude" else "planner"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("search_logs", search_logs_node)
    graph.add_node("correlate", correlate_node)
    graph.add_node("reflection", reflection_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("planner")

    graph.add_conditional_edges("planner", route_after_planner, {
        "search_logs": "search_logs",
        "correlate": "correlate",
        "report": "report",
    })

    graph.add_edge("search_logs", "reflection")
    graph.add_edge("correlate", "reflection")

    graph.add_conditional_edges("reflection", route_after_reflection, {
        "planner": "planner",
        "report": "report",
    })

    graph.add_edge("report", END)

    return graph.compile()


def run_investigation(alert_context: str, max_iterations: int = 4) -> dict:
    app = build_graph()
    initial_state = {
        "alert_context": alert_context,
        "messages": [],
        "plan": None,
        "next_action": None,
        "last_params": {},
        "tool_calls_made": [],
        "retrieved_evidence": [],
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "confidence_score": None,
        "final_report": None,
    }
    final_state = app.invoke(initial_state)
    return final_state

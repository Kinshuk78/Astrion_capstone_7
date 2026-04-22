from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    data_loader_node,
    debugger_node,
    detector_node,
    drift_detector_node,
    human_review_node,
    profiler_node,
    ranker_node,
    summariser_node,
)
from .state import TriageState


def _route(state: TriageState) -> str:
    """Deterministic supervisor routing function.

    Pure function — no LLM, no randomness. Attached to START and to every
    worker node except summariser. Returns the name of the next node to run.

    Routing is flag-driven: each node sets its own completion flag to True.
    Pre-setting a flag in the initial state causes the router to skip that node,
    enabling stripped-down evaluation strategies (A, B, C).
    """
    if state.get("error"):
        return END
    if not state["data_loaded"]:
        return "data_loader"
    if not state["metadata_ready"]:
        return "profiler"
    if not state["detection_done"]:
        return "detector"
    if not state["drift_done"]:
        return "drift_detector"
    if not state["debug_done"]:
        return "debugger"
    if state["needs_human_review"] and not state["review_done"]:
        return "human_review"
    if not state["ranking_done"]:
        return "ranker"
    return "summariser"


def build_graph(checkpointer=None, interactive: bool = False):
    """Build and compile the LangGraph triage workflow.

    Graph topology:
        START → [conditional: _route]
            data_loader → [conditional: _route]
            profiler    → [conditional: _route]
            detector    → [conditional: _route]
            drift_detector → [conditional: _route]
            debugger    → [conditional: _route]
            human_review → [conditional: _route]
            ranker      → [conditional: _route]
            summariser  → END

    The routing function is the supervisor: a single deterministic function
    replaces any LLM-based supervisor node.

    Args:
        checkpointer: Must be None. Passing any checkpointer raises RuntimeError
            because DataFrames in state["tables"] are not msgpack-serialisable.
        interactive: Must be False. Raises RuntimeError if True. Set
            ASTRION_AUTO_APPROVE=1 in the environment instead.

    Returns:
        A compiled LangGraph graph ready for invoke() or stream().

    Raises:
        RuntimeError: If checkpointer is not None or interactive is True.
    """
    if checkpointer is not None or interactive:
        raise RuntimeError(
            "build_graph does not support checkpointers. "
            "DataFrames in state['tables'] are not msgpack-serialisable. "
            "For automated runs set ASTRION_AUTO_APPROVE=1 instead of "
            "using interactive=True."
        )

    graph = StateGraph(TriageState)

    graph.add_node("data_loader", data_loader_node)
    graph.add_node("profiler", profiler_node)
    graph.add_node("detector", detector_node)
    graph.add_node("drift_detector", drift_detector_node)
    graph.add_node("debugger", debugger_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("ranker", ranker_node)
    graph.add_node("summariser", summariser_node)

    # Supervisor edges: START and every worker (except summariser) route through _route
    graph.add_conditional_edges(START, _route)
    for node in [
        "data_loader", "profiler", "detector",
        "drift_detector", "debugger", "human_review", "ranker",
    ]:
        graph.add_conditional_edges(node, _route)

    # Summariser always terminates the run
    graph.add_edge("summariser", END)

    return graph.compile(checkpointer=checkpointer)

from __future__ import annotations

from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict


class TriageState(TypedDict):
    """State shared across all LangGraph nodes in the triage graph.

    Each node receives this dict and returns a partial dict updating only the
    keys it modifies. Lists are always returned as complete new values (not
    appended in place) to keep state updates explicit and readable.

    Routing flags (data_loaded, metadata_ready, ...) drive the deterministic
    supervisor routing function _route() in workflow.py. Setting a flag to True
    in the initial state will cause the router to skip that node.
    """
    # Run configuration
    source: str           # "clean" | "injected"
    sensitivity: str      # "normal" | "high"

    # Data (held in-memory — DataFrames are not serialised to checkpoint)
    tables: Dict[str, Any]      # name → pd.DataFrame
    table_sizes: Dict[str, int]
    db_path: str

    # Agent outputs as plain dicts (serialisable to JSON/checkpoint)
    metadata: Dict[str, Any]     # name → TableMeta as dict
    raw_issues: List[Dict]       # QualityIssue dicts from detector
    drift_issues: List[Dict]     # QualityIssue dicts from drift_detector
    all_issues: List[Dict]       # merged: raw + drift
    verified_issues: List[Dict]  # VerifiedIssue dicts from debugger
    ranked_issues: List[Dict]    # RankedIssue dicts from ranker

    # Supervisor routing flags (each node sets its own flag to True)
    data_loaded: bool
    metadata_ready: bool
    detection_done: bool
    drift_done: bool
    debug_done: bool
    review_done: bool
    ranking_done: bool

    # Human-in-the-loop
    needs_human_review: bool
    human_decision: Optional[str]

    # Observability
    agent_trace: List[str]       # node names in execution order
    timing: Dict[str, float]     # node_name → elapsed seconds
    report_md: str
    error: Optional[str]


def initial_state(source: str = "injected", sensitivity: str = "normal") -> TriageState:
    """Return a zeroed TriageState for the start of a new triage run.

    Callers running stripped-down strategies (e.g., evaluation strategy A)
    can pre-set skip flags in the returned dict before passing to graph.invoke().
    """
    return {  # type: ignore[return-value]
        "source": source,
        "sensitivity": sensitivity,
        "tables": {},
        "table_sizes": {},
        "db_path": "",
        "metadata": {},
        "raw_issues": [],
        "drift_issues": [],
        "all_issues": [],
        "verified_issues": [],
        "ranked_issues": [],
        "data_loaded": False,
        "metadata_ready": False,
        "detection_done": False,
        "drift_done": False,
        "debug_done": False,
        "review_done": False,
        "ranking_done": False,
        "needs_human_review": False,
        "human_decision": None,
        "agent_trace": [],
        "timing": {},
        "report_md": "",
        "error": None,
    }

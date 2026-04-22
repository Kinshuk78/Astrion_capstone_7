"""Generate Astrion DQ architecture diagram as JPEG."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

fig, ax = plt.subplots(figsize=(22, 16))
ax.set_xlim(0, 22)
ax.set_ylim(0, 16)
ax.axis("off")
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

# ── Color palette ──────────────────────────────────────────────────────────
C = {
    "bg":       "#0d1117",
    "panel":    "#161b22",
    "border":   "#30363d",
    "blue":     "#1f6feb",
    "green":    "#238636",
    "orange":   "#d29922",
    "red":      "#da3633",
    "purple":   "#8b5cf6",
    "teal":     "#0e7490",
    "pink":     "#ec4899",
    "text":     "#e6edf3",
    "muted":    "#8b949e",
    "arrow":    "#58a6ff",
    "langgraph":"#f97316",
}

def box(ax, x, y, w, h, color, label, sublabel=None, radius=0.3, alpha=1.0, emoji=""):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.5,
        edgecolor=color,
        facecolor=color + "22" if alpha < 1 else color + "18",
        zorder=3,
    )
    ax.add_patch(patch)
    # top colour bar
    bar = FancyBboxPatch(
        (x, y + h - 0.28), w, 0.28,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=0,
        edgecolor="none",
        facecolor=color + "55",
        zorder=4,
    )
    ax.add_patch(bar)
    ax.text(x + w/2, y + h/2 + (0.15 if sublabel else 0), label,
            ha="center", va="center", color=C["text"],
            fontsize=9.5, fontweight="bold", zorder=5)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.28, sublabel,
                ha="center", va="center", color=C["muted"],
                fontsize=7.5, zorder=5)

def arrow(ax, x1, y1, x2, y2, label="", color=None, style="->"):
    col = color or C["arrow"]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=col,
                                lw=1.8, connectionstyle="arc3,rad=0.0"),
                zorder=6)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.08, my + 0.08, label, color=col,
                fontsize=7, ha="left", zorder=7)

def curved_arrow(ax, x1, y1, x2, y2, rad=0.15, label="", color=None):
    col = color or C["arrow"]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.8,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=6)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.08, my, label, color=col, fontsize=7, ha="left", zorder=7)

def section_label(ax, x, y, text, color):
    ax.text(x, y, text, color=color, fontsize=8, fontweight="bold",
            ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color+"22",
                      edgecolor=color+"66", linewidth=1),
            zorder=8)

# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
ax.text(11, 15.5, "Astrion DQ  —  System Architecture  (v0.5.0)",
        ha="center", va="center", color=C["text"],
        fontsize=16, fontweight="bold")
ax.text(11, 15.1, "Workflow: LangGraph Multi-Agent Triage  |  Pattern: Supervisor + Worker Nodes  |  Strategy A / B / C",
        ha="center", va="center", color=C["muted"], fontsize=9)

# ══════════════════════════════════════════════════════════════════════════════
# ZONE 1 — ENTRY POINTS  (top-left)
# ══════════════════════════════════════════════════════════════════════════════
# Background panel
entry_panel = FancyBboxPatch((0.3, 11.8), 4.2, 2.8,
    boxstyle="round,pad=0,rounding_size=0.3",
    linewidth=1, edgecolor=C["border"], facecolor=C["panel"], zorder=1)
ax.add_patch(entry_panel)
section_label(ax, 0.5, 14.75, "  ENTRY POINTS  ", C["blue"])

box(ax, 0.6, 13.2, 1.7, 0.9, C["blue"], "CLI", "astrion-dq triage", emoji="⌨")
box(ax, 2.6, 13.2, 1.7, 0.9, C["purple"], "REST API", "POST /triage", emoji="🌐")
box(ax, 0.6, 11.9, 1.7, 0.9, C["teal"], "Dashboard", "Streamlit", emoji="📊")
box(ax, 2.6, 11.9, 1.7, 0.9, C["green"], "Evaluate", "A/B/C strategies", emoji="🔬")

# ══════════════════════════════════════════════════════════════════════════════
# ZONE 2 — LANGGRAPH SUPERVISOR  (centre column)
# ══════════════════════════════════════════════════════════════════════════════
sup_panel = FancyBboxPatch((5.0, 3.5), 7.5, 11.0,
    boxstyle="round,pad=0,rounding_size=0.4",
    linewidth=2, edgecolor=C["langgraph"]+"88", facecolor=C["panel"], zorder=1)
ax.add_patch(sup_panel)
section_label(ax, 5.2, 14.65, "  LANGGRAPH StateGraph  —  Supervisor + Worker Pattern  ", C["langgraph"])

# Supervisor routing node
box(ax, 6.2, 13.3, 3.2, 0.95, C["langgraph"], "_route() Supervisor",
    "Deterministic flag-driven routing", emoji="🔀")

# Worker nodes (vertical stack)
nodes = [
    ("data_loader",    C["blue"],   "Data Loader",     "CSV → DuckDB",           "📥", 11.9),
    ("profiler",       C["teal"],   "Profiler",        "infer_metadata()",        "🔍", 10.6),
    ("detector",       C["red"],    "Detector",        "5 checks in parallel",    "⚡", 9.3),
    ("drift_detector", C["orange"], "Drift Detector",  "PSI + KS test",           "📈", 8.0),
    ("debugger",       C["purple"], "Debugger",        "SQL cross-validation",    "🐛", 6.7),
    ("human_review",   C["pink"],   "Human Review",    "interrupt() / auto-approve","👤", 5.5),
    ("ranker",         C["green"],  "Ranker",          "BIS v2 scoring",          "🏆", 4.3),
]
node_y = {}
for nid, col, label, sub, emoji, y in nodes:
    box(ax, 5.4, y, 6.8, 0.9, col, label, sub, emoji=emoji)
    node_y[nid] = y + 0.45

# Summariser
box(ax, 5.4, 3.7, 6.8, 0.9, C["teal"], "Summariser", "LLM exec summary + MD report", emoji="📝")
node_y["summariser"] = 4.15

# Supervisor arrows (left side)
sx = 6.2
for i, (nid, _, _, _, _, y) in enumerate(nodes):
    from_y = 13.3 if i == 0 else nodes[i-1][5] + 0.45
    to_y   = y + 0.45
    if i == 0:
        arrow(ax, 6.55, 13.3, 6.55, 12.8, color=C["langgraph"])
    else:
        arrow(ax, 6.55, nodes[i-1][5], 6.55, y + 0.9, color=C["langgraph"])
arrow(ax, 6.55, 5.5, 6.55, 5.2, color=C["langgraph"])  # human_review → ranker row
arrow(ax, 6.55, 4.3, 6.55, 4.6, color=C["langgraph"])
arrow(ax, 6.55, 4.3, 6.55, 3.7 + 0.9, color=C["langgraph"])

# Right-side feedback arrow (routing loop)
ax.annotate("", xy=(12.2, 13.6), xytext=(12.2, 4.15),
    arrowprops=dict(arrowstyle="<->", color=C["langgraph"]+"88", lw=1.5,
                    connectionstyle="arc3,rad=0.0"), zorder=6)
ax.text(12.35, 8.8, "_route()\nafter each\nnode", color=C["langgraph"],
        fontsize=7, ha="left", va="center", rotation=90)

# START / END labels
ax.text(7.75, 14.55, "START", color=C["langgraph"], fontsize=8.5,
        fontweight="bold", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f9731622",
                  edgecolor=C["langgraph"], lw=1.5))
ax.text(7.75, 3.3, "END", color=C["langgraph"], fontsize=8.5,
        fontweight="bold", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f9731622",
                  edgecolor=C["langgraph"], lw=1.5))

# ══════════════════════════════════════════════════════════════════════════════
# ZONE 3 — DATA LAYER  (bottom-left)
# ══════════════════════════════════════════════════════════════════════════════
data_panel = FancyBboxPatch((0.3, 3.5), 4.2, 7.8,
    boxstyle="round,pad=0,rounding_size=0.4",
    linewidth=1, edgecolor=C["border"], facecolor=C["panel"], zorder=1)
ax.add_patch(data_panel)
section_label(ax, 0.5, 11.45, "  DATA LAYER  ", C["blue"])

box(ax, 0.6, 10.1, 3.6, 0.9, C["blue"],  "Raw CSV Files", "data/raw/retail/", emoji="📂")
box(ax, 0.6, 8.8,  3.6, 0.9, C["blue"],  "Injected CSV",  "data/injected/retail/", emoji="💉")
box(ax, 0.6, 7.5,  3.6, 0.9, C["teal"],  "DuckDB",        "retail.duckdb  (in-memory)", emoji="🦆")
box(ax, 0.6, 6.2,  3.6, 0.9, C["orange"],"Drift Snapshots","outputs/drift_snapshots/", emoji="📸")
box(ax, 0.6, 4.9,  3.6, 0.9, C["orange"],"Schema Snapshots","outputs/schema_snapshots/", emoji="🗂")
box(ax, 0.6, 3.65, 3.6, 0.9, C["green"], "Run Log",       "outputs/run_log.jsonl", emoji="📋")

# arrows: CSV → DuckDB
arrow(ax, 2.4, 10.1, 2.4, 8.4, color=C["blue"])
arrow(ax, 2.4, 8.8, 2.4, 8.4, color=C["blue"])
arrow(ax, 2.4, 8.4, 2.4, 7.5 + 0.9, color=C["blue"])

# ══════════════════════════════════════════════════════════════════════════════
# ZONE 4 — TOOLS & LIBS  (right column)
# ══════════════════════════════════════════════════════════════════════════════
tools_panel = FancyBboxPatch((13.2, 3.5), 8.5, 11.0,
    boxstyle="round,pad=0,rounding_size=0.4",
    linewidth=1, edgecolor=C["border"], facecolor=C["panel"], zorder=1)
ax.add_patch(tools_panel)
section_label(ax, 13.4, 14.65, "  TOOLS · PROTOCOLS · APIs  ", C["purple"])

# ── Python stack ──
section_label(ax, 13.5, 13.9, "Python Stack", C["muted"])
box(ax, 13.5, 12.85, 3.7, 0.8, C["blue"],   "LangGraph 1.1.6",   "StateGraph / TypedDict State")
box(ax, 17.5, 12.85, 3.7, 0.8, C["blue"],   "FastAPI 0.110",     "REST  HTTP/1.1  JSON")
box(ax, 13.5, 11.75, 3.7, 0.8, C["teal"],   "Streamlit",         "WebSocket / HTTP")
box(ax, 17.5, 11.75, 3.7, 0.8, C["teal"],   "Typer",             "CLI  argparse-based")
box(ax, 13.5, 10.65, 3.7, 0.8, C["orange"], "DuckDB",            "SQL  columnar analytics")
box(ax, 17.5, 10.65, 3.7, 0.8, C["orange"], "Pandas / NumPy",    "DataFrame processing")
box(ax, 13.5, 9.55,  3.7, 0.8, C["red"],    "SciPy stats",       "KS-2samp  PSI")
box(ax, 17.5, 9.55,  3.7, 0.8, C["red"],    "WeasyPrint",        "HTML → PDF report")

# ── LLM / external APIs ──
section_label(ax, 13.5, 9.1, "LLM / External APIs", C["muted"])
box(ax, 13.5, 8.0,  3.7, 0.8, C["purple"],  "OpenRouter API",    "https://openrouter.ai/api/v1")
box(ax, 17.5, 8.0,  3.7, 0.8, C["purple"],  "OpenAI SDK",        "Wire format  (chat.completions)")
box(ax, 13.5, 6.9,  3.7, 0.8, C["pink"],    "Claude Sonnet 4.6", "anthropic/claude-sonnet-4-6")
box(ax, 17.5, 6.9,  3.7, 0.8, C["muted"],   "Fallback",          "Template summary (no key)")

# ── Auth & config ──
section_label(ax, 13.5, 6.45, "Auth & Config", C["muted"])
box(ax, 13.5, 5.35, 3.7, 0.8, C["green"],  "Bearer Token",       "ASTRION_API_TOKEN (env)")
box(ax, 17.5, 5.35, 3.7, 0.8, C["green"],  "python-dotenv",      "config/.env loader")

# ── Evaluation ──
section_label(ax, 13.5, 4.9, "Evaluation Metrics", C["muted"])
box(ax, 13.5, 3.8,  3.7, 0.8, C["teal"],   "Precision / Recall", "F1  ·  Noise  ·  Wall time")
box(ax, 17.5, 3.8,  3.7, 0.8, C["orange"], "Strategy A/B/C",     "Baseline/Supervisor/Full")

# ══════════════════════════════════════════════════════════════════════════════
# ZONE 5 — OUTPUTS  (bottom-right zone, below graph)
# ══════════════════════════════════════════════════════════════════════════════
out_panel = FancyBboxPatch((0.3, 0.5), 21.4, 2.7,
    boxstyle="round,pad=0,rounding_size=0.4",
    linewidth=1, edgecolor=C["border"], facecolor=C["panel"], zorder=1)
ax.add_patch(out_panel)
section_label(ax, 0.5, 3.35, "  OUTPUTS  ", C["green"])

out_items = [
    ("ranked_issues\n*.json", C["green"],  0.6,  1.0, 2.4, 1.5),
    ("triage_report\n*.md / *.pdf", C["teal"], 3.2, 1.0, 2.4, 1.5),
    ("run_log.jsonl\nAudit trail",   C["orange"], 5.8, 1.0, 2.4, 1.5),
    ("evaluation_comparison\n*.json",C["blue"], 8.4, 1.0, 2.4, 1.5),
    ("drift_snapshots\n*.json",      C["orange"],11.0, 1.0, 2.4, 1.5),
    ("schema_snapshots\n*.json",     C["purple"],13.6, 1.0, 2.4, 1.5),
    ("Streamlit\nDashboard  :8501",  C["teal"],  16.2, 1.0, 2.4, 1.5),
    ("FastAPI\nREST  :8000",         C["purple"],18.8, 1.0, 2.4, 1.5),
]
for label, col, x, y, w, h in out_items:
    box(ax, x, y, w, h, col, label)

# ══════════════════════════════════════════════════════════════════════════════
# CROSS-ZONE ARROWS
# ══════════════════════════════════════════════════════════════════════════════

# Entry → Supervisor (right arrows)
arrow(ax, 4.3, 13.65, 5.4, 13.65, color=C["arrow"])  # CLI → graph
arrow(ax, 4.3, 12.35, 5.4, 12.35, color=C["arrow"])  # REST API → graph
arrow(ax, 4.3, 12.35, 5.4, 12.35, color=C["arrow"])

# Dashboard → REST API label
ax.text(2.45, 12.85, "HTTP /triage", color=C["muted"], fontsize=7, ha="center")

# Data layer → data_loader
arrow(ax, 4.2, 10.55, 5.4, 10.55, color=C["blue"])  # DuckDB → graph interior (loader)
arrow(ax, 4.2, 6.65, 5.4, 6.65, color=C["orange"])  # snapshots → drift_detector

# Graph → Outputs
arrow(ax, 7.75, 3.5, 7.75, 2.5, color=C["green"])   # summariser → output area

# API → REST box
arrow(ax, 2.6 + 1.7/2, 11.9, 2.6 + 1.7/2, 11.0, color=C["purple"])

# OpenRouter ↔ summariser
curved_arrow(ax, 13.5, 8.4, 12.2, 8.4, rad=-0.1, label="HTTPS REST", color=C["purple"])

# ══════════════════════════════════════════════════════════════════════════════
# DETECTION CHECKS CALLOUT  (beside detector node)
# ══════════════════════════════════════════════════════════════════════════════
checks = [
    "① Null / Missing values",
    "② Duplicate rows",
    "③ Numeric outliers (IQR)",
    "④ Invalid future dates",
    "⑤ RI breaks (FK joins)",
]
cx, cy = 13.0, 9.05
ax.text(cx, cy + 0.25, "Parallel Checks (ThreadPool)", color=C["red"],
        fontsize=7.5, fontweight="bold", ha="left")
for i, c in enumerate(checks):
    ax.text(cx, cy - 0.18*i, c, color=C["muted"], fontsize=7, ha="left")
arrow(ax, 12.2, 9.2, 13.0, 9.2, color=C["red"])

# BIS formula callout
ax.text(13.0, 4.85, "BIS = base_weight × evidence_density\n"
        "       × report_criticality × severity_mult",
        color=C["green"], fontsize=7, ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#23863622",
                  edgecolor=C["green"]+"66", lw=1))
arrow(ax, 12.2, 4.75, 13.0, 4.75, color=C["green"])

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    (C["langgraph"], "LangGraph StateGraph"),
    (C["blue"],      "Data / CLI layer"),
    (C["purple"],    "API / LLM layer"),
    (C["green"],     "Output / Ranking"),
    (C["orange"],    "Storage / Drift"),
    (C["red"],       "Detection checks"),
    (C["teal"],      "UI / Metadata"),
]
for i, (col, lbl) in enumerate(legend_items):
    x = 13.5 + (i % 4) * 2.1
    y = 0.75 if i < 4 else 0.35
    patch = mpatches.Patch(facecolor=col+"33", edgecolor=col, linewidth=1.5, label=lbl)
    ax.add_patch(FancyBboxPatch((x, y - 0.12), 0.28, 0.24,
                                boxstyle="round,pad=0",
                                facecolor=col+"44", edgecolor=col, lw=1.5, zorder=7))
    ax.text(x + 0.38, y, lbl, color=C["muted"], fontsize=7, va="center", zorder=7)

# ── Footer ────────────────────────────────────────────────────────────────────
ax.text(11, 0.18, "Astrion DQ v0.5.0  ·  Academic Capstone — Evaluating Agentic Workflows for Data Quality Triage  ·  Supervisor: Dr. William So, Synogize",
        ha="center", va="center", color=C["muted"], fontsize=7.5)

plt.tight_layout(pad=0.5)
out = "/Users/jaideepgarlyal/Astrion_capstone_7/docs/architecture.jpg"
plt.savefig(out, format="jpeg", dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"Saved: {out}")

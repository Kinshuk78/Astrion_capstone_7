"""PDF report generator for the Astrion data quality triage system.

Produces a two-page A4 PDF:
  Page 1: Title block, ranked issues table (top 20 by BIS)
  Page 2: Workflow strategy comparison table, agent execution trace

Uses reportlab for layout. No external fonts or assets required.

Two entry points:
  generate_triage_report()       -- writes to disk, returns Path
  generate_triage_report_bytes() -- returns raw bytes (for API streaming,
                                    no disk write required)
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from astrion_dq.config import OUTPUTS_DIR

_DARK = colors.HexColor("#2C3E50")
_LIGHT_ROW = colors.HexColor("#F8F8F8")
_SEV_COLORS = {
    "high":   colors.HexColor("#FFCCCC"),
    "medium": colors.HexColor("#FFF3CC"),
    "low":    colors.HexColor("#CCFFCC"),
}


def _styles():
    base = getSampleStyleSheet()
    base.add(ParagraphStyle("DocTitle",  parent=base["Title"],   fontSize=18, spaceAfter=6))
    base.add(ParagraphStyle("DocSub",    parent=base["Normal"],  fontSize=10,
                             textColor=colors.HexColor("#555555"), spaceAfter=3))
    base.add(ParagraphStyle("Section",   parent=base["Heading2"], fontSize=13, spaceAfter=6))
    base.add(ParagraphStyle("Cell",      parent=base["Normal"],  fontSize=8, leading=10))
    return base


def _title_block(styles) -> Table:
    data = [
        [Paragraph("Astrion Data Quality Triage", styles["DocTitle"])],
        [Paragraph("Evaluating Agentic Workflows for Retail Analytics Warehouses", styles["DocSub"])],
        [Paragraph("Academic Capstone · Dr. William So · Synogize", styles["DocSub"])],
    ]
    t = Table(data, colWidths=[17 * cm])
    t.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    return t


def _issues_table(ranked: List[dict]) -> Table:
    cols = ["#", "Issue ID", "Type", "Table", "Severity", "Evidence\nRows", "BIS"]
    widths = [0.6, 2.5, 3.5, 3.0, 1.8, 2.0, 1.8]
    rows = [cols]
    for i, issue in enumerate(ranked[:20], 1):
        rows.append([
            str(i),
            issue.get("issue_id", ""),
            issue.get("issue_type", ""),
            issue.get("table", ""),
            issue.get("severity", ""),
            str(issue.get("evidence_rows", 0)),
            f"{issue.get('impact_score', 0.0):.4f}",
        ])

    t = Table(rows, colWidths=[w * cm for w in widths])

    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), _DARK),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.grey),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _LIGHT_ROW]),
    ]
    for row_idx, issue in enumerate(ranked[:20], 1):
        sev = issue.get("severity", "low")
        if sev in _SEV_COLORS:
            cmds.append(("BACKGROUND", (4, row_idx), (4, row_idx), _SEV_COLORS[sev]))
    t.setStyle(TableStyle(cmds))
    return t


def _metrics_table(metrics_list: List[dict]) -> Table:
    header = ["Strategy", "Prec.", "Recall", "F1", "Top-5\nRecall", "Noise\nRate", "Sum.\nAcc.", "Wall\n(s)"]
    widths  = [2.8, 1.6, 1.6, 1.5, 1.8, 1.8, 1.8, 1.8]
    rows = [header]
    for m in metrics_list:
        if "error" in m:
            rows.append([m.get("strategy", ""), "ERROR"] + ["—"] * 6)
        else:
            rows.append([
                m.get("strategy", ""),
                f"{m.get('precision', 0):.3f}",
                f"{m.get('recall', 0):.3f}",
                f"{m.get('f1', 0):.3f}",
                f"{m.get('top_5_recall', 0):.3f}",
                f"{m.get('noise_rate', 0):.3f}",
                f"{m.get('summary_accuracy', 0):.3f}",
                f"{m.get('wall_seconds', 0):.1f}",
            ])

    t = Table(rows, colWidths=[w * cm for w in widths])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _DARK),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.grey),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _LIGHT_ROW]),
    ]))
    return t


def _build_story(
    ranked_issues: List[dict],
    metrics_list: Optional[List[dict]],
    agent_trace: Optional[List[str]],
) -> list:
    """Build the reportlab story list shared by both PDF entry points."""
    styles = _styles()
    story = []

    story.append(_title_block(styles))
    story.append(Spacer(1, 0.4 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color=_DARK))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("Ranked Quality Issues", styles["Section"]))
    story.append(Paragraph(
        f"Top {min(20, len(ranked_issues))} issues ordered by V2 Business Impact Score (BIS = "
        "base_weight × severity × log_density × report_criticality).",
        styles["Normal"],
    ))
    story.append(Spacer(1, 0.3 * cm))
    if ranked_issues:
        story.append(_issues_table(ranked_issues))
    else:
        story.append(Paragraph("No issues detected.", styles["Normal"]))

    if metrics_list:
        story.append(PageBreak())
        story.append(Paragraph("Workflow Strategy Comparison", styles["Section"]))
        story.append(Paragraph(
            "A = Baseline (no verification), "
            "B = Supervisor (SQL cross-validation + analyst gate), "
            "C = Full (B + statistical drift detection via PSI and KS test).",
            styles["Normal"],
        ))
        story.append(Spacer(1, 0.3 * cm))
        story.append(_metrics_table(metrics_list))

    if agent_trace:
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph("Agent Execution Trace", styles["Section"]))
        story.append(Paragraph(" → ".join(agent_trace), styles["Normal"]))

    return story


def _doc_kwargs() -> dict:
    return dict(
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )


def generate_triage_report(
    ranked_issues: List[dict],
    metrics_list: Optional[List[dict]] = None,
    agent_trace: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Generate a PDF triage report and write it to disk.

    Args:
        ranked_issues: RankedIssue dicts from ranker_node.
        metrics_list:  Strategy metric dicts from evaluate_all(). Optional.
        agent_trace:   Node execution order list. Optional.
        output_path:   Override output path. Defaults to outputs/triage_report.pdf.

    Returns:
        Path to the generated PDF.
    """
    if output_path is None:
        output_path = OUTPUTS_DIR / "triage_report.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(str(output_path), **_doc_kwargs())
    doc.build(_build_story(ranked_issues, metrics_list, agent_trace))
    return output_path


def generate_triage_report_bytes(
    ranked_issues: List[dict],
    metrics_list: Optional[List[dict]] = None,
    agent_trace: Optional[List[str]] = None,
) -> bytes:
    """Generate a PDF triage report in memory and return raw bytes.

    No disk writes. Used by the FastAPI ``/triage/report.pdf`` endpoint and the
    Streamlit dashboard download button so they work on Render's ephemeral FS.

    Args:
        ranked_issues: RankedIssue dicts from ranker_node.
        metrics_list:  Strategy metric dicts from evaluate_all(). Optional.
        agent_trace:   Node execution order list. Optional.

    Returns:
        Raw PDF bytes.
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, **_doc_kwargs())
    doc.build(_build_story(ranked_issues, metrics_list, agent_trace))
    return buf.getvalue()

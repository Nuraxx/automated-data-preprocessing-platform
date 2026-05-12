"""Preprocessing report generation (JSON / Markdown / PDF).

The Streamlit UI uses this module to generate a detailed report of:
- Dataset overview (before/after)
- Transformations applied
- Methods chosen (imputation/encoding/scaling/outlier removal)
- Data quality scores

PDF generation uses reportlab (pure-Python, Streamlit Cloud-friendly).
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Optional

import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib import colors


def generate_preprocessing_report(
    *,
    generated_at: str,
    app_version: str,
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    raw_quality: Dict[str, Any],
    cleaned_quality: Dict[str, Any],
    column_types: Dict[str, List[str]],
    transformations: List[Dict[str, Any]],
    pipeline_config: Optional[Dict[str, Any]] = None,
    recommendations: Optional[List[str]] = None,
    transformed_dataset_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a structured preprocessing report dict."""

    report: Dict[str, Any] = {
        "meta": {
            "generated_at": generated_at,
            "app_version": app_version,
        },
        "dataset_overview": {
            "raw_shape": {"rows": int(raw_df.shape[0]), "cols": int(raw_df.shape[1])},
            "cleaned_shape": {"rows": int(cleaned_df.shape[0]), "cols": int(cleaned_df.shape[1])},
            "raw_columns": raw_df.columns.tolist(),
            "cleaned_columns": cleaned_df.columns.tolist(),
        },
        "data_quality": {
            "raw": raw_quality,
            "cleaned": cleaned_quality,
        },
        "column_types": column_types,
        "transformations_applied": transformations,
        "pipeline": pipeline_config or {},
        "ai_recommendations": recommendations or [],
        "transformed_dataset": transformed_dataset_info or {},
    }

    return report


def report_to_markdown(report: Dict[str, Any]) -> str:
    """Render report dict to a readable Markdown summary."""

    meta = report.get("meta", {})
    ov = report.get("dataset_overview", {})
    dq = report.get("data_quality", {})

    lines: List[str] = []
    lines.append("# Preprocessing Report")
    lines.append("")
    lines.append(f"**Generated at:** {meta.get('generated_at', '')}")
    lines.append(f"**App version:** {meta.get('app_version', '')}")
    lines.append("")

    raw_shape = ov.get("raw_shape", {})
    cleaned_shape = ov.get("cleaned_shape", {})

    lines.append("## Dataset Overview")
    lines.append(f"- Raw shape: {raw_shape.get('rows', 0)} rows × {raw_shape.get('cols', 0)} columns")
    lines.append(f"- Cleaned shape: {cleaned_shape.get('rows', 0)} rows × {cleaned_shape.get('cols', 0)} columns")
    lines.append("")

    lines.append("## Data Quality Score")
    lines.append(f"- Raw score: {dq.get('raw', {}).get('score', 'N/A')}/100")
    lines.append(f"- Cleaned score: {dq.get('cleaned', {}).get('score', 'N/A')}/100")
    lines.append("")

    lines.append("## Column Types")
    col_types = report.get("column_types", {})
    for k in ["numerical", "categorical", "boolean", "datetime", "other"]:
        cols = col_types.get(k, [])
        lines.append(f"- {k.title()}: {len(cols)}")
    lines.append("")

    lines.append("## Transformations Applied")
    history = report.get("transformations_applied", [])
    if not history:
        lines.append("- (No transformations applied)")
    else:
        for item in history:
            ts = item.get("timestamp", "")
            action = item.get("action", "")
            details = item.get("details", {})
            lines.append(f"- {ts} — **{action}**: {json.dumps(details, ensure_ascii=False)}")
    lines.append("")

    lines.append("## Pipeline Configuration")
    pipe = report.get("pipeline", {})
    if not pipe:
        lines.append("- (Not generated)")
    else:
        for k, v in pipe.items():
            lines.append(f"- **{k}**: {v}")
    lines.append("")

    lines.append("## AI-Style Recommendations")
    recs = report.get("ai_recommendations", [])
    if not recs:
        lines.append("- (No recommendations)")
    else:
        for r in recs:
            lines.append(f"- {r}")

    return "\n".join(lines)


def report_to_json_bytes(report: Dict[str, Any]) -> bytes:
    return json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8")


def report_to_pdf_bytes(report: Dict[str, Any]) -> bytes:
    """Render report into a simple PDF.

    This is intentionally lightweight and text-first to keep Streamlit Cloud
    deployments reliable.
    """

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, title="Preprocessing Report")
    styles = getSampleStyleSheet()

    story: List[Any] = []

    story.append(Paragraph("Preprocessing Report", styles["Title"]))
    story.append(Spacer(1, 12))

    meta = report.get("meta", {})
    story.append(Paragraph(f"Generated at: {meta.get('generated_at', '')}", styles["Normal"]))
    story.append(Paragraph(f"App version: {meta.get('app_version', '')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    ov = report.get("dataset_overview", {})
    raw_shape = ov.get("raw_shape", {})
    cleaned_shape = ov.get("cleaned_shape", {})

    overview_table = Table(
        [
            ["Metric", "Raw", "Cleaned"],
            [
                "Shape (rows × cols)",
                f"{raw_shape.get('rows', 0)} × {raw_shape.get('cols', 0)}",
                f"{cleaned_shape.get('rows', 0)} × {cleaned_shape.get('cols', 0)}",
            ],
        ],
        hAlign="LEFT",
    )

    overview_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ]
        )
    )

    story.append(Paragraph("Dataset Overview", styles["Heading2"]))
    story.append(overview_table)
    story.append(Spacer(1, 12))

    dq = report.get("data_quality", {})
    dq_table = Table(
        [
            ["Quality", "Raw", "Cleaned"],
            ["Score (/100)", str(dq.get("raw", {}).get("score", "")), str(dq.get("cleaned", {}).get("score", ""))],
        ],
        hAlign="LEFT",
    )
    dq_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ]
        )
    )

    story.append(Paragraph("Data Quality", styles["Heading2"]))
    story.append(dq_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Transformations Applied", styles["Heading2"]))
    history = report.get("transformations_applied", [])
    if not history:
        story.append(Paragraph("(No transformations applied)", styles["Normal"]))
    else:
        for item in history[:40]:
            story.append(Paragraph(f"- {item.get('timestamp', '')} — {item.get('action', '')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("AI-Style Recommendations", styles["Heading2"]))
    recs = report.get("ai_recommendations", [])
    if not recs:
        story.append(Paragraph("(No recommendations)", styles["Normal"]))
    else:
        for r in recs[:40]:
            story.append(Paragraph(f"- {r}", styles["Normal"]))

    doc.build(story)
    return buf.getvalue()

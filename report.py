"""
PDF report generator for session summaries.

Produces a professional exportable report from wellness and neuro exam
session data using reportlab (pure Python, no external dependencies beyond pip).
"""

import io
from datetime import datetime, timezone

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# Color palette matching the frontend iOS design
COLOR_TEXT = HexColor("#1C1C1E")
COLOR_SUBTLE = HexColor("#8E8E93")
COLOR_TINT = HexColor("#007AFF")
COLOR_GREEN = HexColor("#34C759")
COLOR_ORANGE = HexColor("#FF9500")
COLOR_RED = HexColor("#FF3B30")
COLOR_SEPARATOR = HexColor("#E5E5EA")


def _disclaimer() -> Paragraph:
    """Return the standard disclaimer paragraph."""
    style = ParagraphStyle("disclaimer", parent=getSampleStyleSheet()["BodyText"],
                           fontSize=8, textColor=COLOR_SUBTLE, alignment=TA_CENTER,
                           spaceBefore=4 * mm, spaceAfter=4 * mm)
    return Paragraph(
        "Wellness indicator, not a medical device. All metrics are relative/qualitative "
        "estimates from signal processing. Not intended for clinical diagnosis.",
        style,
    )


def generate_wellness_report(summary: dict) -> io.BytesIO:
    """
    Generate a PDF wellness session report.

    Args:
        summary: Session summary dict from _build_summary.

    Returns:
        io.BytesIO buffer containing the PDF data.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=20 * mm,
                            bottomMargin=20 * mm, leftMargin=20 * mm,
                            rightMargin=20 * mm)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle("title", parent=styles["Title"],
                                 fontSize=22, textColor=COLOR_TEXT,
                                 spaceAfter=2 * mm)
    story.append(Paragraph("Wellness Monitor — Session Report", title_style))
    story.append(Paragraph(
        datetime.now(timezone.utc).strftime("%B %d, %Y — %H:%M UTC"),
        ParagraphStyle("date", parent=styles["Normal"], fontSize=10,
                       textColor=COLOR_SUBTLE),
    ))
    story.append(Spacer(1, 4 * mm))
    story.append(HRFlowable(width="100%", color=COLOR_SEPARATOR, thickness=1))
    story.append(Spacer(1, 6 * mm))

    # Metrics table
    data = [
        ["Metric", "Value", "Status"],
        [
            "Session Duration",
            f"{summary.get('session_duration_min', '--')} min",
            "—"
        ],
        [
            "Average Heart Rate",
            f"{summary.get('average_heart_rate_bpm', '--')} BPM",
            "Normal" if summary.get('average_heart_rate_bpm') else "Insufficient data"
        ],
        [
            "Average Breathing Rate",
            f"{summary.get('average_breathing_rate_brpm', '--')} br/min",
            "Normal" if summary.get('average_breathing_rate_brpm') else "Insufficient data"
        ],
        [
            "Total Blinks",
            str(summary.get("total_blinks", 0)),
            "—"
        ],
        [
            "No-Face Percentage",
            f"{summary.get('no_face_percentage', 0)}%",
            "—"
        ],
    ]

    table = Table(data, colWidths=[100 * mm, 60 * mm, 35 * mm])
    table_style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#F2F2F7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), COLOR_TEXT),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, COLOR_SEPARATOR),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#FFFFFF"), HexColor("#F9F9F9")]),
    ])
    table.setStyle(table_style)
    story.append(table)

    story.append(Spacer(1, 8 * mm))

    # Drowsiness events
    drowsy = summary.get("drowsiness_events", [])
    if drowsy:
        story.append(Paragraph("Drowsiness Events", styles["Heading2"]))
        for ts, status in drowsy[:10]:
            story.append(Paragraph(
                f"• {ts}s — {status}",
                ParagraphStyle("event", parent=styles["BodyText"],
                               fontSize=9, textColor=COLOR_SUBTLE),
            ))

    story.append(_disclaimer())
    doc.build(story)
    buf.seek(0)
    return buf


def generate_neuro_report(summary: dict) -> io.BytesIO:
    """
    Generate a PDF neurological exam report.

    Args:
        summary: Neuro exam report dict from _build_neuro_report.

    Returns:
        io.BytesIO buffer containing the PDF data.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=20 * mm,
                            bottomMargin=20 * mm, leftMargin=20 * mm,
                            rightMargin=20 * mm)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle("title", parent=styles["Title"],
                                 fontSize=22, textColor=COLOR_TEXT,
                                 spaceAfter=2 * mm)
    story.append(Paragraph("Neurological Motor Exam — Report", title_style))
    story.append(Paragraph(
        datetime.now(timezone.utc).strftime("%B %d, %Y — %H:%M UTC"),
        ParagraphStyle("date", parent=styles["Normal"], fontSize=10,
                       textColor=COLOR_SUBTLE),
    ))
    story.append(Spacer(1, 4 * mm))
    story.append(HRFlowable(width="100%", color=COLOR_SEPARATOR, thickness=1))
    story.append(Spacer(1, 6 * mm))

    # Duration
    story.append(Paragraph(
        f"Exam Duration: {summary.get('exam_duration_s', '--')} seconds",
        styles["BodyText"],
    ))

    # Finger-to-nose
    ftn = summary.get("finger_to_nose")
    if ftn:
        story.append(Paragraph("Finger-to-Nose Test", styles["Heading2"]))
        for side in ["left", "right"]:
            sd = ftn.get(side)
            if not sd:
                continue
            story.append(Paragraph(
                f"{side.title()} — Tremor: {sd.get('tremor_amplitude_px', '--')} px | "
                f"Dysmetria: {sd.get('dysmetria_px', '--')} px | "
                f"Move Time: {sd.get('movement_time_s', '--')} s",
                styles["BodyText"],
            ))
        flags = ftn.get("asymmetry_flags", [])
        if flags:
            for f in flags:
                story.append(Paragraph(f"⚠ {f}", styles["BodyText"]))

    # DDK
    ddk = summary.get("dysdiadochokinesia")
    if ddk:
        story.append(Paragraph("Rapid Hand Movements (Dysdiadochokinesia)", styles["Heading2"]))
        for side, data in ddk.items():
            if isinstance(data, dict):
                story.append(Paragraph(
                    f"{side.title()} — Rate: {data.get('rate_hz', '--')} Hz | "
                    f"Rhythm CV: {data.get('rhythm_cv', '--')} | "
                    f"Decay: {data.get('amplitude_decay_pct', '--')}% | "
                    f"Status: {data.get('status', '--')}",
                    styles["BodyText"],
                ))

    # Romberg
    rom = summary.get("romberg")
    if rom:
        story.append(Paragraph("Romberg Test (Postural Sway)", styles["Heading2"]))
        story.append(Paragraph(
            f"Open Eyes Area: {rom.get('open_ellipse_area', '--')} | "
            f"Closed Eyes Area: {rom.get('closed_ellipse_area', '--')} | "
            f"Romberg Quotient: {rom.get('romberg_quotient', '--')}",
            styles["BodyText"],
        ))
        story.append(Paragraph(
            f"Interpretation: {rom.get('interpretation', rom.get('status', '--'))}",
            ParagraphStyle("interp", parent=styles["BodyText"],
                           textColor=COLOR_TINT, fontSize=11),
        ))

    story.append(_disclaimer())
    doc.build(story)
    buf.seek(0)
    return buf

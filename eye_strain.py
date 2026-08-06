"""
Eye strain risk assessment combining blink rate, screen distance, and exposure.

Screen distance is estimated relative to a one-time calibration measurement,
since a monocular webcam cannot give reliable absolute distance without a
known reference object. Eye strain risk is a composite of low blink rate,
close screen distance, and prolonged exposure duration.

This is a wellness/informational tool, not a medical diagnostic device.
All outputs are relative/qualitative indicators from signal processing.
"""

import time

import numpy as np

from config import (
    DISTANCE_FAR_THRESHOLD,
    DISTANCE_NEAR_THRESHOLD,
    EYE_STRAIN_LOW_BLINK_THRESHOLD,
    EXPOSURE_LOW_THRESHOLD,
    EXPOSURE_HIGH_THRESHOLD,
    LEFT_EYE_OUTER,
    RIGHT_EYE_OUTER,
)


def estimate_screen_distance(landmarks: list[tuple[float, float]],
                             calibration_interocular_px: float) -> dict:
    """
    Estimate relative screen distance using interocular distance ratio.

    A monocular webcam cannot give reliable absolute distance without a known
    reference. This method compares the current interocular pixel distance
    to a calibration measurement taken at a known comfortable distance,
    yielding a relative status.

    Args:
        landmarks: List of (x, y) pixel coords for all face mesh landmarks.
        calibration_interocular_px: Calibrated interocular distance in pixels.

    Returns:
        dict with keys:
            - "status": "closer" | "at_distance" | "farther"
            - "label": Human-readable relative description
            - "ratio": Current-to-calibration ratio
            - "current_px": Current interocular pixel distance
    """
    if len(landmarks) <= max(LEFT_EYE_OUTER, RIGHT_EYE_OUTER):
        return {"status": "at_distance", "label": "Insufficient landmarks",
                "ratio": 1.0, "current_px": 0.0}
    p_left = np.array(landmarks[LEFT_EYE_OUTER])
    p_right = np.array(landmarks[RIGHT_EYE_OUTER])
    current_px = float(np.linalg.norm(p_right - p_left))
    if calibration_interocular_px <= 0:
        return {"status": "at_distance", "label": "Calibration needed",
                "ratio": 1.0, "current_px": current_px}
    ratio = current_px / calibration_interocular_px
    # Larger face (ratio > 1) = closer to screen. Smaller face (ratio < 1) = farther.
    if ratio > DISTANCE_NEAR_THRESHOLD:
        return {"status": "closer", "label": "Closer than calibration distance",
                "ratio": ratio, "current_px": current_px}
    if ratio < DISTANCE_FAR_THRESHOLD:
        return {"status": "farther", "label": "Farther than calibration distance",
                "ratio": ratio, "current_px": current_px}
    return {"status": "at_distance", "label": "At calibration distance",
            "ratio": ratio, "current_px": current_px}


def screen_time_exposure(session_start: float,
                         current_time: float | None = None) -> float:
    """
    Calculate continuous screen exposure minutes in the current session.

    Args:
        session_start: Unix timestamp (seconds) of session start.
        current_time: Current unix timestamp. Defaults to time.time().

    Returns:
        float: Minutes of continuous screen exposure.
    """
    if current_time is None:
        current_time = time.time()
    elapsed = max(0.0, current_time - session_start)
    return elapsed / 60.0


def assess_eye_strain_risk(blink_rate: float,
                           screen_distance_status: str,
                           exposure_minutes: float) -> dict:
    """
    Assess composite eye strain risk from blink rate, distance, and exposure.

    Combines three contributing factors into a risk level with explicit
    explanation of which factor(s) contributed, so the UI can show why.

    Args:
        blink_rate: Blinks per minute (from drowsiness module).
        screen_distance_status: "closer" | "at_distance" | "farther".
        exposure_minutes: Continuous screen exposure in minutes.

    Returns:
        dict with keys:
            - "risk": "low" | "moderate" | "high"
            - "label": Human-readable description
            - "factors": List of contributing factor descriptions
            - "blink_rate": The input blink rate
            - "distance_status": The input distance status
            - "exposure_minutes": The input exposure minutes
    """
    factors = []
    risk_score = 0
    if blink_rate < EYE_STRAIN_LOW_BLINK_THRESHOLD and blink_rate >= 0:
        factors.append("Low blink rate — reduced tear film refresh")
        risk_score += 1
    if screen_distance_status == "closer":
        factors.append("Screen closer than calibration — increased ocular strain")
        risk_score += 1
    if exposure_minutes >= EXPOSURE_HIGH_THRESHOLD:
        factors.append("Over 60 min continuous exposure — take a break")
        risk_score += 2
    elif exposure_minutes >= EXPOSURE_LOW_THRESHOLD:
        factors.append("Over 30 min exposure — consider a break")
        risk_score += 1
    if risk_score >= 3:
        risk = "high"
    elif risk_score >= 1:
        risk = "moderate"
    else:
        risk = "low"
    return {
        "risk": risk,
        "label": f"Eye strain risk: {risk}",
        "factors": factors,
        "blink_rate": blink_rate,
        "distance_status": screen_distance_status,
        "exposure_minutes": exposure_minutes,
    }

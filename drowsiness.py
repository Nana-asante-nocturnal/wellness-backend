"""
Blink detection and drowsiness assessment from eye aspect ratio (EAR).

Uses a 6-point EAR formula per eye (vertical distances / horizontal distance),
a state machine for blink detection (avoids single-frame false positives), and
computes standard drowsiness metrics: blink rate and PERCLOS.

This is a wellness/informational tool, not a medical diagnostic device.
Drowsiness thresholds are heuristic, not clinically validated.
"""

import numpy as np

from config import (
    LEFT_EYE_LANDMARKS,
    RIGHT_EYE_LANDMARKS,
    EAR_THRESHOLD,
    CONSECUTIVE_FRAMES_EYE_CLOSE,
    BLINK_RATE_WINDOW_SEC,
    PERCLOS_WINDOW_SEC,
    BLINK_RATE_LOW,
    BLINK_RATE_DROWSY,
    PERCLOS_MILD_THRESHOLD,
    PERCLOS_DROWSY_THRESHOLD,
)


def eye_aspect_ratio(landmarks: list[tuple[float, float]],
                     eye_side: str) -> float:
    """
    Compute Eye Aspect Ratio (EAR) for a single eye using 6 landmark points.

    Formula: (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    where p1-p4 are the horizontal points and p2,p3,p5,p6 are vertical points.

    Args:
        landmarks: List of (x, y) tuples for all 468 MediaPipe landmarks.
        eye_side: "left" or "right" — selects the correct landmark indices.

    Returns:
        float: Eye Aspect Ratio value. Lower = more closed.
    """
    indices = LEFT_EYE_LANDMARKS if eye_side == "left" else RIGHT_EYE_LANDMARKS
    max_idx = max(indices) if indices else 0
    if len(landmarks) <= max_idx:
        return 0.0
    pts = [landmarks[i] for i in indices]
    if len(pts) < 6:
        return 0.0
    p = np.array(pts)
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    h = np.linalg.norm(p[0] - p[3])
    if h < 1e-6:
        return 0.0
    return float((v1 + v2) / (2.0 * h))


def detect_blink(ear_history: list[float],
                 threshold: float = EAR_THRESHOLD,
                 consecutive_frames: int = CONSECUTIVE_FRAMES_EYE_CLOSE
                 ) -> bool:
    """
    Detect a blink event using a small state machine on EAR history.

    A blink = EAR falls below threshold for at least consecutive_frames frames,
    then recovers above threshold. State machine avoids false-positive flicker
    from single-frame threshold crossings (e.g. noise, landmark jitter).

    Args:
        ear_history: List of recent EAR values (most recent last).
        threshold: EAR value below which the eye is considered closed.
        consecutive_frames: Minimum consecutive frames below threshold.

    Returns:
        bool: True if a blink was just completed on the most recent frame.
    """
    min_len = consecutive_frames + 3
    if len(ear_history) < min_len:
        return False
    if ear_history[-1] < threshold:
        return False
    below_run = 0
    had_above_before = False
    for i in range(len(ear_history) - 2, -1, -1):
        val = ear_history[i]
        if val < threshold:
            below_run += 1
        else:
            if below_run >= consecutive_frames:
                had_above_before = True
            break
    if below_run >= consecutive_frames and had_above_before:
        return True
    return False


def compute_blink_rate(blink_timestamps: list[float],
                       window_seconds: float = BLINK_RATE_WINDOW_SEC) -> float:
    """
    Compute blink rate (blinks per minute) over a trailing time window.

    Args:
        blink_timestamps: List of absolute timestamps (seconds) of blink events.
        window_seconds: Trailing window duration in seconds.

    Returns:
        float: Blinks per minute over the window. 0.0 if no blinks in window.
    """
    if not blink_timestamps or window_seconds <= 0:
        return 0.0
    now = blink_timestamps[-1]
    cutoff = now - window_seconds
    count = sum(1 for t in blink_timestamps if t > cutoff)
    return float(count) / window_seconds * 60.0


def compute_perclos(ear_history: list[float],
                    ear_timestamps: list[float],
                    threshold: float = EAR_THRESHOLD,
                    window_seconds: float = PERCLOS_WINDOW_SEC) -> float:
    """
    Compute PERCLOS — percentage of time eyes were below threshold (closed).

    Standard drowsiness metric used in fatigue research. PERCLOS > 0.3
    is commonly associated with significant drowsiness.

    Args:
        ear_history: List of EAR values synchronized with ear_timestamps.
        ear_timestamps: List of absolute timestamps (seconds) for each EAR value.
        threshold: EAR threshold for eye-closed state.
        window_seconds: Trailing window duration in seconds.

    Returns:
        float: PERCLOS value as a fraction (0.0 to 1.0). 0.0 if insufficient data.
    """
    if len(ear_history) < 2 or len(ear_timestamps) < 2:
        return 0.0
    now = ear_timestamps[-1]
    cutoff = now - window_seconds
    total_frames = 0
    closed_frames = 0
    for ear_val, ts in zip(ear_history, ear_timestamps):
        if ts > cutoff:
            total_frames += 1
            if ear_val < threshold:
                closed_frames += 1
    if total_frames == 0:
        return 0.0
    return float(closed_frames) / float(total_frames)


def assess_drowsiness(blink_rate: float, perclos: float) -> dict:
    """
    Assess drowsiness level from blink rate and PERCLOS using heuristic thresholds.

    Thresholds are heuristic, not clinically validated. Based on published
    driver-fatigue research suggesting correlations between elevated blink
    rate/PERCLOS and drowsiness, but individual baselines vary.

    Args:
        blink_rate: Blinks per minute (from compute_blink_rate).
        perclos: PERCLOS value as a fraction (from compute_perclos).

    Returns:
        dict with keys:
            - "status": "alert" | "mild_fatigue" | "drowsy"
            - "label": Human-readable description
            - "blink_rate": The input blink rate
            - "perclos": The input PERCLOS value
    """
    if perclos >= PERCLOS_DROWSY_THRESHOLD:
        return {
            "status": "drowsy",
            "label": "High PERCLOS — eyes closed for significant portion of window; may be drowsy",
            "blink_rate": blink_rate,
            "perclos": perclos,
        }
    if perclos >= PERCLOS_MILD_THRESHOLD or blink_rate >= BLINK_RATE_DROWSY:
        return {
            "status": "mild_fatigue",
            "label": "Elevated PERCLOS or blink rate — possible mild fatigue",
            "blink_rate": blink_rate,
            "perclos": perclos,
        }
    return {
        "status": "alert",
        "label": "Normal blink rate and eye-open percentage — appears alert",
        "blink_rate": blink_rate,
        "perclos": perclos,
    }

"""
Unit tests for drowsiness.py — eye aspect ratio, blink detection, PERCLOS.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from drowsiness import (
    eye_aspect_ratio,
    detect_blink,
    compute_blink_rate,
    compute_perclos,
    assess_drowsiness,
)


def _make_eye_landmarks(openness: float) -> list:
    """Build synthetic face landmarks with configurable eye openness."""
    landmarks = [(0, 0)] * 468
    eye_w = 30.0
    eye_h = 10.0 * openness
    left_pts = [(100 + eye_w, 150), (100, 150 - eye_h), (100 + eye_w * 0.5, 150 - eye_h),
                (100, 150), (100 + eye_w * 0.5, 150 + eye_h), (100, 150 + eye_h)]
    right_pts = [(200 + eye_w, 150), (200, 150 - eye_h), (200 + eye_w * 0.5, 150 - eye_h),
                 (200, 150), (200 + eye_w * 0.5, 150 + eye_h), (200, 150 + eye_h)]
    from config import LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS
    for idx_src, idx_dst in enumerate(LEFT_EYE_LANDMARKS):
        landmarks[idx_dst] = left_pts[idx_src]
    for idx_src, idx_dst in enumerate(RIGHT_EYE_LANDMARKS):
        landmarks[idx_dst] = right_pts[idx_src]
    return landmarks


def test_eye_aspect_ratio_open():
    """EAR should be higher when eyes are open."""
    landmarks = _make_eye_landmarks(1.0)
    ear_left = eye_aspect_ratio(landmarks, "left")
    ear_right = eye_aspect_ratio(landmarks, "right")
    assert ear_left > 0.2
    assert ear_right > 0.2


def test_eye_aspect_ratio_closed():
    """EAR should be lower (near zero) when eyes are closed."""
    landmarks = _make_eye_landmarks(0.01)
    ear_left = eye_aspect_ratio(landmarks, "left")
    assert ear_left < 0.05


def test_detect_blink_no_blink_when_eyes_open():
    """Should not detect blink when EAR stays above threshold."""
    ear_history = [0.35, 0.36, 0.34, 0.35, 0.36, 0.34]
    assert not detect_blink(ear_history)


def test_detect_blink_state_machine():
    """Should detect blink only after closure and recovery sequence."""
    ear_history = [0.35, 0.35, 0.15, 0.10, 0.12, 0.38]
    assert detect_blink(ear_history)


def test_compute_blink_rate_empty():
    """Should return 0.0 with no blink timestamps."""
    assert compute_blink_rate([]) == 0.0


def test_compute_blink_rate_calculation():
    """Should compute correct blinks per minute."""
    now = 100.0
    timestamps = [now - 50, now - 30, now - 10, now - 5]
    rate = compute_blink_rate(timestamps, window_seconds=60)
    assert rate > 0.0
    assert rate == 4.0


def test_compute_perclos():
    """PERCLOS should reflect fraction of frames below threshold."""
    ear_history = [0.35] * 20 + [0.10] * 10
    ear_ts = list(np.arange(len(ear_history)) * (1.0 / 30.0))
    perclos = compute_perclos(ear_history, ear_ts, window_seconds=2.0)
    assert perclos > 0.3
    assert perclos < 0.4


def test_assess_drowsiness_alert():
    """Normal blink_rate and low PERCLOS should return alert."""
    result = assess_drowsiness(12.0, 0.05)
    assert result["status"] == "alert"


def test_assess_drowsiness_drowsy():
    """High PERCLOS should return drowsy."""
    result = assess_drowsiness(15.0, 0.35)
    assert result["status"] == "drowsy"

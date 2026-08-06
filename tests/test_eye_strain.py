"""
Unit tests for eye_strain.py — screen distance estimation and strain risk.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from eye_strain import (
    estimate_screen_distance,
    screen_time_exposure,
    assess_eye_strain_risk,
)


def _make_landmarks(interocular_px: float = 100.0):
    """Build synthetic landmarks with configurable interocular distance."""
    landmarks = [(0, 0)] * 468
    landmarks[33] = (0.0, 150.0)
    landmarks[263] = (interocular_px, 150.0)
    return landmarks


def test_estimate_screen_distance_closer():
    """Should detect 'closer' when current distance is much larger than calibration."""
    landmarks = _make_landmarks(130.0)
    result = estimate_screen_distance(landmarks, 100.0)
    assert result["status"] == "closer"


def test_estimate_screen_distance_farther():
    """Should detect 'farther' when current distance is much smaller."""
    landmarks = _make_landmarks(70.0)
    result = estimate_screen_distance(landmarks, 100.0)
    assert result["status"] == "farther"


def test_estimate_screen_distance_at_calibration():
    """Should detect 'at_distance' when close to calibration."""
    landmarks = _make_landmarks(105.0)
    result = estimate_screen_distance(landmarks, 100.0)
    assert result["status"] == "at_distance"


def test_estimate_screen_distance_no_calibration():
    """Should handle zero calibration value gracefully."""
    landmarks = _make_landmarks(100.0)
    result = estimate_screen_distance(landmarks, 0.0)
    assert result["status"] == "at_distance"


def test_screen_time_exposure():
    """Should calculate correct exposure minutes from session start."""
    start = time.time() - 120
    exposure = screen_time_exposure(start)
    assert 1.8 < exposure < 2.2


def test_assess_eye_strain_low_risk():
    """Low risk when all factors are normal."""
    result = assess_eye_strain_risk(15.0, "at_distance", 10.0)
    assert result["risk"] == "low"


def test_assess_eye_strain_high_risk():
    """High risk when multiple factors are poor."""
    result = assess_eye_strain_risk(5.0, "closer", 70.0)
    assert result["risk"] == "high"


def test_assess_eye_strain_moderate_risk():
    """Moderate risk with one contributing factor."""
    result = assess_eye_strain_risk(5.0, "at_distance", 40.0)
    assert result["risk"] == "moderate"

"""
Unit tests for hrv_stress.py — IBI, RMSSD, and stress scoring logic.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hrv_stress import compute_ibi, compute_rmssd, rmssd_to_stress_score


def test_compute_ibi_empty_for_insufficient_peaks():
    """Should return empty list with fewer than 2 peaks."""
    assert compute_ibi([]) == []
    assert compute_ibi([1.0]) == []


def test_compute_ibi_correct_intervals():
    """Should compute correct inter-beat intervals in milliseconds."""
    peak_times = [0.0, 0.85, 1.7, 2.55, 3.4]
    ibis = compute_ibi(peak_times)
    assert len(ibis) == 4
    assert all(abs(ibi - 850.0) < 5.0 for ibi in ibis)


def test_compute_rmssd_insufficient_data():
    """Should return None with fewer than 5 IBIs."""
    assert compute_rmssd([]) is None
    assert compute_rmssd([800, 820, 810, 780]) is None


def test_compute_rmssd_known_values():
    """RMSSD should match hand-calculated value for known IBIs."""
    ibis = [800, 810, 820, 830, 840, 850]
    rmssd = compute_rmssd(ibis)
    assert rmssd is not None
    expected_sq_diffs = [(810-800)**2, (820-810)**2, (830-820)**2, (840-830)**2, (850-840)**2]
    expected = (sum(expected_sq_diffs) / 5) ** 0.5
    assert abs(rmssd - expected) < 0.01


def test_rmssd_to_stress_calibrating():
    """Should return 'calibrating' status when no baseline exists."""
    result = rmssd_to_stress_score(50.0, None)
    assert result["status"] == "calibrating"
    assert result["baseline_rmssd"] is None


def test_rmssd_to_stress_lower():
    """Should detect 'lower_than_baseline' when RMSSD drops significantly."""
    result = rmssd_to_stress_score(30.0, 60.0)
    assert result["status"] == "lower_than_baseline"


def test_rmssd_to_stress_higher():
    """Should detect 'higher_than_baseline' when RMSSD rises significantly."""
    result = rmssd_to_stress_score(90.0, 60.0)
    assert result["status"] == "higher_than_baseline"


def test_rmssd_to_stress_within_normal():
    """Should detect 'within_normal' when RMSSD is close to baseline."""
    result = rmssd_to_stress_score(55.0, 60.0)
    assert result["status"] == "within_normal"

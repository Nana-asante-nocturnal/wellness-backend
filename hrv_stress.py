"""
Heart Rate Variability (HRV) and relative stress scoring from pulse peak times.

Computes inter-beat intervals (IBI), RMSSD (a standard time-domain HRV metric),
and translates RMSSD into a relative stress score compared to a personal baseline.

IMPORTANT: This is a wellness/informational tool, not a medical diagnostic device.
RMSSD baselines vary significantly person to person — absolute universal thresholds
would be misleading. All stress scores are intentionally relative, comparing current
RMSSD to a user's own calibrated baseline.
"""

import numpy as np


def compute_ibi(peak_times: list[float]) -> list[float]:
    """
    Compute inter-beat intervals from successive pulse peak timestamps.

    Args:
        peak_times: List of timestamps (seconds) of detected pulse peaks,
                    monotonically increasing.

    Returns:
        list[float]: Inter-beat intervals in milliseconds. Empty list if < 2 peaks.
    """
    if len(peak_times) < 2:
        return []
    ibis = []
    for i in range(1, len(peak_times)):
        ibi_s = peak_times[i] - peak_times[i - 1]
        ibis.append(ibi_s * 1000.0)
    return ibis


def compute_rmssd(ibi_list: list[float]) -> float | None:
    """
    Compute RMSSD — root mean square of successive IBI differences.

    RMSSD is a standard time-domain HRV metric reflecting parasympathetic
    (vagal) activity. Lower RMSSD is associated with higher physiological
    stress in research literature (documented correlation, not causation).

    Args:
        ibi_list: List of inter-beat intervals in milliseconds.

    Returns:
        float: RMSSD in milliseconds, or None if fewer than 5 IBIs available.
    """
    if len(ibi_list) < 5:
        return None
    diffs = []
    for i in range(1, len(ibi_list)):
        diffs.append(ibi_list[i] - ibi_list[i - 1])
    if not diffs:
        return None
    squared = [d * d for d in diffs]
    return float(np.sqrt(np.mean(squared)))


def rmssd_to_stress_score(rmssd: float,
                          baseline_rmssd: float | None = None) -> dict:
    """
    Convert RMSSD to a relative stress status using personal baseline comparison.

    This is intentionally relative, not absolute, because RMSSD baselines vary
    significantly person to person. An absolute universal threshold would be
    misleading as a "stress measurement."

    Args:
        rmssd: Current RMSSD value in milliseconds.
        baseline_rmssd: User's calibrated baseline RMSSD, or None if unavailable.

    Returns:
        dict with keys:
            - "status": "calibrating" | "lower_than_baseline" | "within_normal" |
                        "higher_than_baseline"
            - "label": Human-readable relative description
            - "rmssd": The current RMSSD value
            - "baseline_rmssd": The baseline value or None
    """
    if baseline_rmssd is None:
        return {
            "status": "calibrating",
            "label": "Baseline still calibrating — collect 90s of resting data",
            "rmssd": rmssd,
            "baseline_rmssd": None,
        }
    ratio = rmssd / baseline_rmssd if baseline_rmssd > 0 else 1.0
    if ratio < 0.75:
        status = "lower_than_baseline"
        label = "Lower RMSSD than your baseline — may indicate elevated stress"
    elif ratio > 1.25:
        status = "higher_than_baseline"
        label = "Higher RMSSD than your baseline — may indicate relaxation"
    else:
        status = "within_normal"
        label = "RMSSD within normal range of your baseline"
    return {
        "status": status,
        "label": label,
        "rmssd": rmssd,
        "baseline_rmssd": baseline_rmssd,
    }

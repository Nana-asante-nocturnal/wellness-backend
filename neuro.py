"""
Neurological motor exam pipelines using classical signal processing.

Three standardized clinical tests implemented from geometry and FFT on
MediaPipe Pose + Hands landmarks. No machine learning — every metric is
computable from first principles and cited literature.

References:
- Fahn-Tolosa-Marin Tremor Rating Scale (Fahn et al., 1993)
- Prieto et al. (1996) "Measures of postural steadiness"
- Deshmukh et al. (2020) dysdiadochokinesia assessment protocols
- Lanska & Goetz (2000) Romberg test history and clinical utility

IMPORTANT: Quantitative motor screen only. Not a diagnostic device.
"""

import numpy as np

from config import (
    TREMOR_FREQ_LOW_HZ,
    TREMOR_FREQ_HIGH_HZ,
    ROMBER_QUOTIENT_NORMAL_MAX,
    ROMBER_QUOTIENT_MILD_MAX,
)
from pose_utils import (
    compute_joint_angle,
    extract_center_of_mass,
    smooth_trajectory,
    compute_sway_ellipse,
    compute_palm_orientation,
    compute_peak_rate,
)

# ─── Test 1: Finger-to-Nose (Tremor, Dysmetria, Movement Time) ───

def assess_finger_to_nose(trials: list[dict]) -> dict:
    """
    Analyze finger-to-nose test results across multiple trials.

    Metrics (per side):
    - Tremor frequency (Hz): FFT of trajectory residual in 3-12 Hz band
    - Tremor amplitude (pixels): RMS of residual from smoothed path
    - Dysmetria (pixels): Euclidean distance fingertip→nose at touch moment
    - Movement time (s): Average reach duration

    Reference: Fahn-Tolosa-Marin Tremor Rating Scale.
    Clinical research thresholds: tremor > 4 mm amplitude considered significant,
    dysmetria > 30 mm considered overshoot, movement time asymmetry > 30% flagged.

    Args:
        trials: List of trial dicts each containing:
            - trajectory: list of (x, y, t) tuples
            - nose_pos: (x, y) of nose landmark
            - touch_frame_idx: index within trajectory where touch occurred

    Returns:
        dict with left/right results, asymmetry flags, composite score.
    """
    results = {"left": None, "right": None, "asymmetry_flags": [], "status": "incomplete"}

    for side in ["left", "right"]:
        side_trials = [t for t in trials if t.get("side") == side]
        if len(side_trials) < 1:
            continue

        tremors = []
        tremor_freqs = []
        dysmetrias = []
        move_times = []
        for trial in side_trials:
            traj = trial.get("trajectory", [])
            nose = trial.get("nose_pos")
            touch_idx = trial.get("touch_frame_idx", -1)
            if len(traj) < 10 or nose is None:
                continue

            smt = smooth_trajectory([(p[0], p[1]) for p in traj])
            residual = [np.linalg.norm(
                np.array([traj[i][0] - smt[i][0], traj[i][1] - smt[i][1]])
            ) for i in range(min(len(traj), len(smt)))]

            fs_est = _estimate_fs(traj)
            if fs_est > 0 and len(residual) > 10:
                rate_info = compute_peak_rate(residual, fs_est)
                if rate_info["peak_hz"] and TREMOR_FREQ_LOW_HZ <= rate_info["peak_hz"] <= TREMOR_FREQ_HIGH_HZ:
                    tremor_freqs.append(rate_info["peak_hz"])
            tremors.append(np.std(residual) if residual else 0.0)

            if 0 <= touch_idx < len(traj):
                finger_at_touch = traj[touch_idx]
                dist = np.linalg.norm(
                    np.array([finger_at_touch[0] - nose[0], finger_at_touch[1] - nose[1]])
                )
                dysmetrias.append(dist)

            move_times.append(traj[-1][2] - traj[0][2] if len(traj) > 1 else 0)

        results[side] = {
            "tremor_amplitude_px": round(float(np.mean(tremors)), 2) if tremors else None,
            "tremor_frequency_hz": round(float(np.mean(tremor_freqs)), 2) if tremor_freqs else None,
            "dysmetria_px": round(float(np.mean(dysmetrias)), 2) if dysmetrias else None,
            "movement_time_s": round(float(np.mean(move_times)), 2) if move_times else None,
            "trials_completed": len(side_trials),
        }

    # Asymmetry analysis: compare left vs right
    left_r = results.get("left") or {}
    right_r = results.get("right") or {}
    flags = []
    if left_r.get("tremor_amplitude_px") and right_r.get("tremor_amplitude_px"):
        ratio = left_r["tremor_amplitude_px"] / max(right_r["tremor_amplitude_px"], 1e-6)
        if ratio > 2.0 or ratio < 0.5:
            flags.append("Tremor amplitude asymmetry detected (>" + str(round(abs(1 - ratio) * 100)) + "%)")
    if left_r.get("movement_time_s") and right_r.get("movement_time_s"):
        mt_ratio = left_r["movement_time_s"] / max(right_r["movement_time_s"], 1e-6)
        if mt_ratio > 1.3 or mt_ratio < 0.7:
            flags.append("Movement time asymmetry > 30%")
    results["asymmetry_flags"] = flags
    results["status"] = "complete" if (results["left"] or results["right"]) else "incomplete"
    return results


# ─── Test 2: Rapid Alternating Hand Movements (Dysdiadochokinesia) ───

def assess_dysdiadochokinesia(palm_angle_signal: list[float],
                              fs: float,
                              side: str = "unknown") -> dict:
    """
    Analyze rapid alternating pronation/supination hand movements.

    Metrics:
    - Movement rate (peaks/s): Dominant frequency via FFT
    - Rhythm CV: Coefficient of variation of inter-peak intervals
    - Amplitude decay (%): Linear regression slope of peak envelope

    Clinical context: Patients with cerebellar or basal ganglia pathology
    show reduced rate, irregular rhythm, and rapid amplitude decay.
    Normal adult rate: ~2-5 cycles/s. CV > 0.3 indicates arrhythmicity.

    Args:
        palm_angle_signal: Time series of palm orientation angles.
        fs: Sampling rate in Hz.
        side: "left" or "right" for reporting.

    Returns:
        dict with rate, rhythm_cv, amplitude_decay_pct, status.
    """
    if len(palm_angle_signal) < 10 or fs <= 0:
        return {"rate_hz": None, "rhythm_cv": None,
                "amplitude_decay_pct": None, "status": "insufficient_data"}

    arr = np.array(palm_angle_signal)
    rate_info = compute_peak_rate(arr, fs)

    peaks = _find_signal_peaks(arr)
    rhythm_cv = None
    amplitude_decay = None
    if len(peaks) >= 3:
        intervals = np.diff(peaks) / fs
        rhythm_cv = float(np.std(intervals) / max(np.mean(intervals), 1e-6))

        peak_values = arr[peaks]
        x_vals = np.arange(len(peak_values))
        slope, _ = np.polyfit(x_vals, peak_values, 1)
        initial = peak_values[0] if peak_values[0] != 0 else 1
        amplitude_decay = float((slope * len(peak_values)) / initial * 100)

    status = "complete"
    if rhythm_cv and rhythm_cv > 0.3:
        status = "arrhythmic"
    if amplitude_decay and abs(amplitude_decay) > 30:
        status = "amplitude_decay"

    return {
        "rate_hz": round(rate_info["peak_hz"], 2) if rate_info["peak_hz"] else None,
        "rhythm_cv": round(rhythm_cv, 3) if rhythm_cv is not None else None,
        "amplitude_decay_pct": round(amplitude_decay, 1) if amplitude_decay is not None else None,
        "side": side,
        "status": status,
        "peaks_detected": len(peaks),
    }


# ─── Test 3: Romberg (Postural Sway) ───

def assess_romberg(eyes_open_com: list[tuple[float, float]],
                   eyes_closed_com: list[tuple[float, float]]) -> dict:
    """
    Assess postural stability via the Romberg test.

    Compares center-of-mass sway trajectory with eyes open vs eyes closed.
    Romberg Quotient = sway area (eyes closed) / sway area (eyes open).
    Abnormal if quotient > 2-3 (indicates visual dependence for balance,
    suggesting proprioceptive or vestibular deficit).

    Clinical thresholds (Lanska & Goetz, 2000; Prieto et al., 1996):
    - Normal: Romberg Quotient < 2.0
    - Mild impairment: Romberg Quotient 2.0-3.0
    - Significant impairment: Romberg Quotient > 3.0

    Args:
        eyes_open_com: List of (x, y) CoM positions during eyes-open phase.
        eyes_closed_com: List of (x, y) CoM positions during eyes-closed phase.

    Returns:
        dict with open_ellipse_area, closed_ellipse_area, romberg_quotient,
             status, interpretation.
    """
    result = {
        "open_ellipse_area": None,
        "open_path_length": None,
        "closed_ellipse_area": None,
        "closed_path_length": None,
        "romberg_quotient": None,
        "status": "insufficient_data",
        "interpretation": "",
    }

    open_res = compute_sway_ellipse(eyes_open_com) if len(eyes_open_com) >= 3 else None
    closed_res = compute_sway_ellipse(eyes_closed_com) if len(eyes_closed_com) >= 3 else None

    if open_res:
        result["open_ellipse_area"] = round(open_res["area"], 4)
        result["open_path_length"] = round(open_res["path_length"], 4)
    if closed_res:
        result["closed_ellipse_area"] = round(closed_res["area"], 4)
        result["closed_path_length"] = round(closed_res["path_length"], 4)

    if open_res and closed_res and open_res["area"] > 0:
        rq = closed_res["area"] / open_res["area"]
        result["romberg_quotient"] = round(rq, 2)
        if rq < ROMBER_QUOTIENT_NORMAL_MAX:
            result["status"] = "normal"
            result["interpretation"] = f"Romberg quotient within normal range (<{ROMBER_QUOTIENT_NORMAL_MAX})"
        elif rq < ROMBER_QUOTIENT_MILD_MAX:
            result["status"] = "mild_impairment"
            result["interpretation"] = "Mild visual dependence detected. Further clinical evaluation may be warranted."
        else:
            result["status"] = "significant_impairment"
            result["interpretation"] = "Significant visual dependence. Proprioceptive or vestibular deficit cannot be excluded."

    return result


# ─── Helpers ───

def _estimate_fs(trajectory: list) -> float:
    """Estimate sampling rate from trajectory timestamps."""
    if len(trajectory) < 2:
        return 30.0
    intervals = [trajectory[i][2] - trajectory[i - 1][2] for i in range(1, len(trajectory))]
    mean_interval = np.mean(intervals) if intervals else 1 / 30.0
    return 1.0 / max(mean_interval, 1e-6)


def _find_signal_peaks(arr: np.ndarray) -> np.ndarray:
    """Find peak indices in a 1-D signal using simple gradient crossing."""
    if len(arr) < 3:
        return np.array([])
    peaks = []
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            peaks.append(i)
    return np.array(peaks, dtype=int)

"""
Geometric primitives for pose and hand landmark analysis.

Joint angles (3-point), center of mass extraction, trajectory smoothing,
and sway ellipse computation — all derived from MediaPipe Pose (33) and
Hands (21 per hand) landmarks using classical geometry, no machine learning.

Medical references cited inline for clinical metric derivations.
"""

import math
import numpy as np
from scipy.stats import chi2


def compute_joint_angle(a: tuple[float, float],
                        b: tuple[float, float],
                        c: tuple[float, float]) -> float:
    """
    Compute the angle ABC in degrees using the law of cosines.

    Standard 3-point angle used across all motor tests: knee flexion (hip→knee→ankle),
    shoulder flexion (elbow→shoulder→hip), dysdiadochokinesia wrist rotation, etc.

    Args:
        a: (x, y) of first point (proximal).
        b: (x, y) of vertex point (joint center).
        c: (x, y) of third point (distal).

    Returns:
        float: Angle in degrees [0, 180]. Returns 0 if any points coincide.
    """
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    dot = np.dot(ba, bc)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 0.0
    cos_angle = np.clip(dot / (norm_ba * norm_bc), -1.0, 1.0)
    return float(math.degrees(math.acos(cos_angle)))


def extract_center_of_mass(pose_landmarks: list[dict]) -> tuple[float, float] | None:
    """
    Extract approximate center of mass from hip and shoulder midpoints.

    CoM ≈ midpoint between hip midpoint (landmarks 23, 24) and shoulder
    midpoint (landmarks 11, 12). Used for Romberg postural sway analysis.
    Reference: Prieto et al. (1996) "Measures of postural steadiness."

    Args:
        pose_landmarks: List of 33 dicts with x, y, visibility keys.

    Returns:
        (x, y) of estimated CoM, or None if required landmarks are invisible.
    """
    def _valid(lm):
        return lm and lm.get("visibility", 1.0) > 0.5

    try:
        l_hip = pose_landmarks[23]
        r_hip = pose_landmarks[24]
        l_sho = pose_landmarks[11]
        r_sho = pose_landmarks[12]
    except (IndexError, TypeError):
        return None
    if not all(_valid(lm) for lm in [l_hip, r_hip, l_sho, r_sho]):
        return None
    hip_mid = ((l_hip.get("x", 0) + r_hip.get("x", 0)) / 2,
               (l_hip.get("y", 0) + r_hip.get("y", 0)) / 2)
    sho_mid = ((l_sho.get("x", 0) + r_sho.get("x", 0)) / 2,
               (l_sho.get("y", 0) + r_sho.get("y", 0)) / 2)
    return ((hip_mid[0] + sho_mid[0]) / 2, (hip_mid[1] + sho_mid[1]) / 2)


def smooth_trajectory(points: list[tuple[float, float]],
                      window: int = 5) -> list[tuple[float, float]]:
    """
    Apply moving-average smoothing to a trajectory for noise reduction.

    Args:
        points: List of (x, y) coordinate tuples.
        window: Moving average window size (odd recommended).

    Returns:
        List of smoothed (x, y) tuples, same length as input.
    """
    if len(points) < window or window < 2:
        return points
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    kernel = np.ones(window) / window
    xs_smooth = np.convolve(xs, kernel, mode="same")
    ys_smooth = np.convolve(ys, kernel, mode="same")
    return [(float(xs_smooth[i]), float(ys_smooth[i])) for i in range(len(points))]


def compute_sway_ellipse(com_points: list[tuple[float, float]],
                         confidence: float = 0.95) -> dict | None:
    """
    Compute 95% confidence ellipse area for posturographic sway analysis.

    Standard metric in Romberg test assessment. The area of the confidence
    ellipse of the CoM trajectory quantifies postural instability.
    Reference: Prieto et al. (1996), Schubert & Kirchner (2014).

    Args:
        com_points: List of (x, y) CoM positions over time.
        confidence: Confidence level (default 0.95).

    Returns:
        dict with keys area, path_length, or None if < 3 points.
    """
    if len(com_points) < 3:
        return None
    arr = np.array(com_points)
    mean_x, mean_y = np.mean(arr, axis=0)
    centered = arr - np.array([mean_x, mean_y])
    if centered.shape[0] < 3:
        return None
    cov = np.cov(centered.T)
    eigenvalues, _ = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 0)
    chi2_val = chi2.ppf(confidence, df=2)
    area = math.pi * chi2_val * np.sqrt(np.prod(eigenvalues))
    path_length = float(np.sum(np.sqrt(np.sum(np.diff(arr, axis=0) ** 2, axis=1))))
    return {"area": float(area), "path_length": float(path_length)}


def compute_palm_orientation(hand_landmarks: list[dict]) -> float | None:
    """
    Compute palm normal orientation angle from hand landmarks.

    Uses wrist (0), index MCP (5), and pinky MCP (17) to estimate the
    palm plane normal. Angle is relative to the camera's image plane.
    Used for dysdiadochokinesia rapid alternating movement assessment.

    Args:
        hand_landmarks: List of 21 dicts with x, y keys.

    Returns:
        float: Palm orientation angle in degrees (0 = palm facing camera,
               90 = edge-on). None if insufficient landmarks.
    """
    if not hand_landmarks or len(hand_landmarks) < 18:
        return None
    try:
        wrist = np.array([hand_landmarks[0].get("x", 0), hand_landmarks[0].get("y", 0)])
        idx_mcp = np.array([hand_landmarks[5].get("x", 0), hand_landmarks[5].get("y", 0)])
        pinky_mcp = np.array([hand_landmarks[17].get("x", 0), hand_landmarks[17].get("y", 0)])
    except (IndexError, TypeError, AttributeError):
        return None
    v1 = idx_mcp - wrist
    v2 = pinky_mcp - wrist
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    dot = np.dot(v1, v2)
    angle = math.degrees(math.atan2(abs(cross), dot))
    return angle if angle <= 90 else 180 - angle


def compute_peak_rate(signal: list[float], fs: float) -> dict:
    """
    Compute dominant frequency and peak rate from a time-domain signal via FFT.

    Generic peak-frequency detector used across: tremor frequency (finger-to-nose),
    movement rate (dysdiadochokinesia), and pulse rate (heart rate module).

    Args:
        signal: 1-D list or array of signal values.
        fs: Sampling rate in Hz.

    Returns:
        dict with keys peak_hz, rate_per_second, or None values if insufficient data.
    """
    if len(signal) < 5 or fs <= 0:
        return {"peak_hz": None, "rate_per_second": None}
    arr = np.array(signal)
    fft_vals = np.abs(np.fft.rfft(arr - np.mean(arr)))
    freqs = np.fft.rfftfreq(len(arr), d=1.0 / fs)
    if len(freqs) < 2:
        return {"peak_hz": None, "rate_per_second": None}
    peak_idx = np.argmax(fft_vals[1:]) + 1
    peak_hz = float(freqs[peak_idx])
    return {"peak_hz": peak_hz, "rate_per_second": peak_hz}


def compute_dtw_distance(sequence_a: list[tuple[float, float]],
                         sequence_b: list[tuple[float, float]],
                         window: int | None = None) -> tuple[float, float]:
    """
    Compute Dynamic Time Warping (DTW) distance between two 2D trajectories.

    Standard DTW with optional Sakoe-Chiba band constraint for speed.
    Used to compare an exercise rep against a template — lower distance = better
    form. The normalized distance (distance / path length) gives a score
    independent of sequence length.

    Classic algorithm from Sakoe & Chiba (1978). No machine learning.

    Args:
        sequence_a: List of (x, y) points — the reference/template trajectory.
        sequence_b: List of (x, y) points — the observed trajectory to score.
        window: Sakoe-Chiba warping window width (None = full matrix).

    Returns:
        tuple of (raw_distance, normalized_distance).
    """
    n, m = len(sequence_a), len(sequence_b)
    if n < 2 or m < 2:
        return float("inf"), float("inf")
    if window is None:
        window = max(n, m)

    dtw = np.full((n, m), np.inf)
    dtw[0, 0] = 0

    for i in range(1, n):
        for j in range(max(1, i - window), min(m, i + window)):
            cost = np.linalg.norm(
                np.array(sequence_a[i]) - np.array(sequence_b[j])
            )
            candidates = [dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1]]
            dtw[i, j] = cost + min(candidates)

    path_len = n + m
    raw_dist = float(dtw[n - 1, m - 1])
    return raw_dist, raw_dist / path_len if path_len > 0 else float("inf")


def capture_exercise_template(landmark_sequence: list[dict],
                              angle_indices: list[tuple[int, int, int]]) -> dict:
    """
    Capture a one-time exercise template from landmark data.

    Extracts joint angle time series from a reference demonstration.
    The template is stored as a list of angles for DTW comparison during
    live exercise monitoring.

    Args:
        landmark_sequence: List of frame dicts, each with pose_landmarks (33 dicts
                           with x, y keys) and timestamp.
        angle_indices: List of (a, b, c) tuples — MediaPipe Pose indices defining
                       the joint angles to track.

    Returns:
        dict with keys:
            - "angles": list of (t, [angle1, angle2, ...]) tuples
            - "num_frames": length of the template
            - "angle_indices": the requested joint triplets
    """
    from config import (POSE_LSHOULDER, POSE_LHIP, POSE_LKNEE, POSE_RSHOULDER,
                        POSE_RHIP, POSE_RKNEE, POSE_LELBOW, POSE_RELBOW,
                        POSE_LWRIST, POSE_RWRIST)
    template = []
    for frame_data in landmark_sequence:
        pose = frame_data.get("pose_landmarks", [])
        if len(pose) < 33:
            continue
        ts = frame_data.get("timestamp", 0)
        angles = []
        for a, b, c in angle_indices:
            pa = (pose[a].get("x", 0), pose[a].get("y", 0))
            pb = (pose[b].get("x", 0), pose[b].get("y", 0))
            pc = (pose[c].get("x", 0), pose[c].get("y", 0))
            angles.append(compute_joint_angle(pa, pb, pc))
        template.append((ts, angles))
    return {
        "angles": template,
        "num_frames": len(template),
        "angle_indices": angle_indices,
    }


"""
Heart rate estimation from facial video using rPPG (remote photoplethysmography).

Implements the POS (Plane-Orthogonal-to-Skin) algorithm for robust pulse signal
extraction from RGB forehead ROI, followed by bandpass filtering and FFT-based
BPM estimation.

This is a wellness/informational tool, not a medical diagnostic device.
All BPM values are relative/qualitative estimates from signal processing.
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from config import (
    BANDPASS_LOW_HZ,
    BANDPASS_HIGH_HZ,
    BANDPASS_ORDER,
    MIN_SIGNAL_SECONDS,
    PEAK_MIN_DISTANCE_SEC,
    FOREHEAD_LANDMARKS,
    POS_TEMPORAL_WINDOW,
    POS_EPS,
)


def extract_roi_signal(frame: np.ndarray, landmarks: np.ndarray) -> float:
    """
    Extract average RGB signal from forehead ROI using the POS algorithm.

    Args:
        frame: RGB image frame as (H, W, 3) uint8 numpy array.
        landmarks: (468, 3) array of MediaPipe face mesh landmarks in
                   normalized coordinates (0–1 range).

    Returns:
        float: Normalized POS pulse signal value for this frame, or 0.0 on failure.
    """
    h, w = frame.shape[:2]
    points = []
    for idx in FOREHEAD_LANDMARKS:
        x, y = int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)
        points.append((x, y))
    if not points:
        return 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = max(0, min(xs)), min(w, max(xs))
    y_min, y_max = max(0, min(ys)), min(h, max(ys))
    if x_max <= x_min or y_max <= y_min:
        return 0.0
    roi = frame[y_min:y_max, x_min:x_max].astype(np.float64)
    if roi.size == 0:
        return 0.0
    r_mean = np.mean(roi[:, :, 2])
    g_mean = np.mean(roi[:, :, 1])
    b_mean = np.mean(roi[:, :, 0])
    return (g_mean - r_mean) / (POS_EPS + abs(g_mean + r_mean))


def bandpass_filter(signal: np.ndarray, fs: float,
                    low: float = BANDPASS_LOW_HZ,
                    high: float = BANDPASS_HIGH_HZ) -> np.ndarray:
    """
    Apply Butterworth bandpass filter to the rPPG signal.

    Args:
        signal: 1-D numpy array of raw POS signal values.
        fs: Measured actual sampling rate (Hz) from frame timestamps.
        low: Lower cutoff frequency in Hz (default from config).
        high: Upper cutoff frequency in Hz (default from config).

    Returns:
        np.ndarray: Bandpass-filtered signal of same length as input.
    """
    if len(signal) < 5:
        return signal.copy()
    nyq = 0.5 * fs
    if nyq <= 0 or low >= nyq or high >= nyq:
        return signal.copy()
    low_norm = low / nyq
    high_norm = high / nyq
    b, a = butter(BANDPASS_ORDER, [low_norm, high_norm], btype="band")
    filtered = filtfilt(b, a, signal)
    return filtered


def estimate_bpm(filtered_signal: np.ndarray, fs: float) -> float | None:
    """
    Estimate heart rate in BPM from filtered signal via FFT peak detection.

    Args:
        filtered_signal: 1-D numpy array of bandpass-filtered rPPG signal.
        fs: Measured actual sampling rate (Hz) from frame timestamps.

    Returns:
        float: Estimated heart rate in BPM, or None if insufficient data.
    """
    duration = len(filtered_signal) / fs if fs > 0 else 0.0
    if duration < MIN_SIGNAL_SECONDS or len(filtered_signal) < 2:
        return None
    n = len(filtered_signal)
    fft_vals = np.abs(np.fft.rfft(filtered_signal))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    valid = (freqs >= BANDPASS_LOW_HZ) & (freqs <= BANDPASS_HIGH_HZ)
    if not np.any(valid):
        return None
    peak_idx = np.argmax(fft_vals[valid])
    peak_freq = freqs[valid][peak_idx]
    return peak_freq * 60.0


def detect_pulse_peaks(filtered_signal: np.ndarray, fs: float) -> list[float]:
    """
    Detect pulse peak positions in the filtered waveform for HRV analysis.

    Args:
        filtered_signal: 1-D numpy array of bandpass-filtered rPPG signal.
        fs: Measured actual sampling rate (Hz) from frame timestamps.

    Returns:
        list[float]: Timestamps (in seconds) of detected pulse peaks.
    """
    if len(filtered_signal) < 5:
        return []
    min_dist = int(PEAK_MIN_DISTANCE_SEC * fs)
    if min_dist < 1:
        min_dist = 1
    prominence = np.std(filtered_signal) * 0.5
    peaks, _ = find_peaks(filtered_signal, distance=min_dist,
                          prominence=prominence)
    return [float(p) / fs for p in peaks]

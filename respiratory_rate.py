"""
Respiratory rate estimation from the same rPPG (POS) signal used for heart rate.

The POS signal carries both cardiac (~0.7-4.0 Hz) and respiratory (~0.1-0.5 Hz)
modulations. A second bandpass filter isolates the respiratory component, and
FFT peak detection estimates breaths per minute.

Reference: VitalLens approach (Rouast-Labs), pyVHR respiratory module.
Classical signal processing — no deep learning required.

Wellness/informational tool — not a medical diagnostic device.
"""

import numpy as np
from scipy.signal import butter, filtfilt
from config import RESP_BANDPASS_LOW_HZ, RESP_BANDPASS_HIGH_HZ, BANDPASS_ORDER, MIN_SIGNAL_SECONDS_RESP


def bandpass_filter_resp(signal: np.ndarray, fs: float,
                         low: float = RESP_BANDPASS_LOW_HZ,
                         high: float = RESP_BANDPASS_HIGH_HZ) -> np.ndarray:
    """
    Apply Butterworth bandpass filter tuned for respiratory frequency range.

    Respiratory rate in adults is typically 12-20 breaths/min (0.2-0.33 Hz).
    The wider 0.1-0.5 Hz band accounts for slow/deep breathing (6 bpm) and
    rapid breathing (up to 30 bpm).

    Args:
        signal: 1-D numpy array of raw POS signal values.
        fs: Measured actual sampling rate (Hz) from frame timestamps.
        low: Lower cutoff frequency in Hz (default from config).
        high: Upper cutoff frequency in Hz (default from config).

    Returns:
        np.ndarray: Bandpass-filtered respiratory signal of same length as input.
    """
    if len(signal) < 10:
        return signal.copy()
    nyq = 0.5 * fs
    if nyq <= 0 or low >= nyq:
        return signal.copy()
    low_norm = min(low / nyq, 0.99)
    high_norm = min(high / nyq, 0.99)
    if low_norm >= high_norm or low_norm <= 0:
        return signal.copy()
    b, a = butter(BANDPASS_ORDER, [low_norm, high_norm], btype="band")
    filtered = filtfilt(b, a, signal)
    return filtered


def estimate_breathing_rate(filtered_signal: np.ndarray, fs: float) -> float | None:
    """
    Estimate breathing rate in breaths per minute from filtered signal via FFT.

    Uses FFT peak detection in the respiratory band (0.1-0.5 Hz).
    Requires at least MIN_SIGNAL_SECONDS_RESP of data before returning a value.

    Args:
        filtered_signal: 1-D numpy array of bandpass-filtered respiratory signal.
        fs: Measured actual sampling rate (Hz) from frame timestamps.

    Returns:
        float: Estimated breathing rate in breaths/min, or None if insufficient data.
    """
    duration = len(filtered_signal) / fs if fs > 0 else 0.0
    if duration < MIN_SIGNAL_SECONDS_RESP or len(filtered_signal) < 10:
        return None
    n = len(filtered_signal)
    fft_vals = np.abs(np.fft.rfft(filtered_signal))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    valid = (freqs >= RESP_BANDPASS_LOW_HZ) & (freqs <= RESP_BANDPASS_HIGH_HZ)
    if not np.any(valid) or np.max(fft_vals[valid]) < 1e-8:
        return None
    peak_idx = np.argmax(fft_vals[valid])
    peak_freq = freqs[valid][peak_idx]
    return peak_freq * 60.0

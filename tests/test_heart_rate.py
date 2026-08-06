"""
Unit tests for heart_rate.py using synthetic signals.

Tests bandpass filter behavior and FFT-based BPM estimation accuracy
with a known-frequency sine wave to verify the pipeline returns correct BPM.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from numpy.random import RandomState
from heart_rate import bandpass_filter, estimate_bpm, detect_pulse_peaks

RNG = RandomState(42)


def test_bandpass_filter_preserves_length():
    """Bandpass filter output should have same length as input."""
    fs = 30.0
    t = np.arange(0, 10, 1 / fs)
    signal = np.sin(2 * np.pi * 1.0 * t)
    filtered = bandpass_filter(signal, fs)
    assert len(filtered) == len(signal)


def test_bandpass_filter_attenuates_out_of_band():
    """Filter should attenuate frequencies outside 0.7-4.0 Hz."""
    fs = 30.0
    t = np.arange(0, 15, 1 / fs)
    signal_0_3hz = np.sin(2 * np.pi * 0.3 * t)
    signal_10hz = np.sin(2 * np.pi * 10.0 * t)
    filtered_low = bandpass_filter(signal_0_3hz, fs)
    filtered_high = bandpass_filter(signal_10hz, fs)
    rms_low = np.sqrt(np.mean(filtered_low ** 2))
    rms_high = np.sqrt(np.mean(filtered_high ** 2))
    rms_original = np.sqrt(np.mean(signal_0_3hz ** 2))
    assert rms_low < rms_original * 0.5
    assert rms_high < np.sqrt(np.mean(signal_10hz ** 2)) * 0.5


def test_estimate_bpm_with_known_frequency():
    """Synthetic sine at 1.2 Hz (72 BPM) should return approximately 72 BPM."""
    fs = 30.0
    duration = 15.0
    t = np.arange(0, duration, 1 / fs)
    signal = np.sin(2 * np.pi * 1.2 * t)
    filtered = bandpass_filter(signal, fs)
    bpm = estimate_bpm(filtered, fs)
    assert bpm is not None
    assert abs(bpm - 72.0) < 3.0


def test_estimate_bpm_insufficient_data():
    """Should return None when signal duration is too short."""
    fs = 30.0
    t = np.arange(0, 5, 1 / fs)
    signal = np.sin(2 * np.pi * 1.2 * t)
    filtered = bandpass_filter(signal, fs)
    bpm = estimate_bpm(filtered, fs)
    assert bpm is None


def test_detect_pulse_peaks_finds_peaks():
    """Should detect peaks in a clean sinusoidal signal at known frequency."""
    fs = 30.0
    duration = 12.0
    t = np.arange(0, duration, 1 / fs)
    signal = np.sin(2 * np.pi * 1.2 * t)
    filtered = bandpass_filter(signal, fs)
    peaks = detect_pulse_peaks(filtered, fs)
    assert len(peaks) > 0
    if len(peaks) >= 2:
        intervals = np.diff(peaks)
        avg_interval = np.mean(intervals)
        expected_interval = 1.0 / 1.2
        assert abs(avg_interval - expected_interval) < 0.3

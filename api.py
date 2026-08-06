"""
FastAPI + WebSocket backend for the wellness monitor.

Receives landmark data from the browser (MediaPipe runs client-side to avoid
bandwidth issues with raw video), processes through all four wellness modules,
and pushes results back in real time.

Wellness/informational tool — not a medical diagnostic device.
"""

import asyncio
import json
import subprocess
import sys
import time
from datetime import datetime
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from config import (
    CORS_ORIGINS,
    SIGNAL_BUFFER_SECONDS,
    WS_HEARTBEAT_SEC,
    WS_MAX_FRAME_AGE_SEC,
    CALIBRATION_DURATION_SEC,
)
from heart_rate import (
    bandpass_filter,
    estimate_bpm,
    detect_pulse_peaks,
)
from respiratory_rate import bandpass_filter_resp, estimate_breathing_rate
from hrv_stress import compute_ibi, compute_rmssd, rmssd_to_stress_score
from drowsiness import (
    eye_aspect_ratio,
    detect_blink,
    compute_blink_rate,
    compute_perclos,
    assess_drowsiness,
)
from eye_strain import (
    estimate_screen_distance,
    screen_time_exposure,
    assess_eye_strain_risk,
)
from history import init_db, save_session, list_sessions, get_session, delete_session
from neuro import assess_finger_to_nose, assess_dysdiadochokinesia, assess_romberg
from pose_utils import compute_palm_orientation, extract_center_of_mass
from report import generate_wellness_report, generate_neuro_report


app = FastAPI(title="Wellness Monitor API", version="0.1.0")
_startup_time = time.time()
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """Initialize database tables on server start."""
    init_db()


class SessionState:
    """Holds per-session rolling buffers and calibration data."""

    def __init__(self) -> None:
        self.raw_signals: list[float] = []
        self.filtered_signals: list[float] = []
        self.timestamps: list[float] = []
        self.ear_history: list[float] = []
        self.ear_timestamps: list[float] = []
        self.blink_timestamps: list[float] = []
        self.peak_times: list[float] = []
        self.fs: float = 30.0
        # Calibration
        self.baseline_rmssd: float | None = None
        self.calibration_interocular_px: float = 0.0
        self.calibration_data: list[dict] = []
        self.calibrating: bool = False
        self.calibration_start: float = 0.0
        # Session
        self.session_start: float = time.time()
        self.frame_count: int = 0
        self.no_face_count: int = 0
        # Summary data for end-of-session
        self.bpm_history: list[tuple[float, float]] = []
        self.rmssd_history: list[tuple[float, float]] = []
        self.drowsiness_events: list[tuple[float, str]] = []
        self.eye_strain_history: list[tuple[float, str]] = []
        self.resp_signal: list[float] = []
        self.resp_history: list[tuple[float, float]] = []

    def trim_buffers(self) -> None:
        """Remove data older than signal buffer window."""
        if not self.timestamps:
            return
        cutoff = self.timestamps[-1] - SIGNAL_BUFFER_SECONDS
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.pop(0)
            if self.raw_signals:
                self.raw_signals.pop(0)
            if self.filtered_signals:
                self.filtered_signals.pop(0)
        while self.ear_timestamps and self.ear_timestamps[0] < cutoff:
            self.ear_timestamps.pop(0)
            if self.ear_history:
                self.ear_history.pop(0)


sessions: dict[str, SessionState] = {}
neuro_sessions: dict[str, "NeuroSessionState"] = {}


class NeuroSessionState:
    """Holds per-session state for the neurological motor exam."""

    def __init__(self) -> None:
        self.current_test: str | None = None
        self.test_phase: str = "idle"
        self.session_start: float = time.time()
        self.phase_start: float = time.time()
        self.frame_count: int = 0
        # Per-test data buffers
        self.trajectories: list[dict] = []
        self.palm_angle_buffer: list[float] = []
        self.angle_timestamps: list[float] = []
        self.com_history: list[tuple[float, float]] = []
        self.com_ts: list[float] = []
        # Accumulated results
        self.finger_nose_results: dict = {}
        self.ddk_results: dict = {}
        self.romberg_results: dict = {}
        self.eyes_open_com: list[tuple[float, float]] = []
        self.eyes_closed_com: list[tuple[float, float]] = []
        self.test_side: str = "right"


def _update_frame_rate(state: SessionState) -> None:
    """Update measured frame rate from timestamp history."""
    if len(state.timestamps) >= 2:
        interval_slice = state.timestamps[-min(30, len(state.timestamps)):]
        intervals = np.diff(interval_slice)
        if len(intervals) > 0:
            state.fs = 1.0 / max(np.mean(intervals), 0.001)


def _process_heart_rate(state: SessionState, ts: float) -> dict:
    """Process heart rate from rolling rPPG signal buffer."""
    bpm = None
    hr_status = "insufficient_data"
    if len(state.raw_signals) >= 10:
        filtered = bandpass_filter(np.array(state.raw_signals), state.fs)
        state.filtered_signals = filtered.tolist()
        bpm = estimate_bpm(np.array(state.filtered_signals), state.fs)
        if bpm is not None:
            hr_status = "ok"
            state.bpm_history.append((ts, bpm))
            peaks = detect_pulse_peaks(np.array(state.filtered_signals), state.fs)
            signal_duration = len(state.filtered_signals) / state.fs
            state.peak_times = [ts - signal_duration + p for p in peaks]
    return {"bpm": round(bpm, 1) if bpm is not None else None, "status": hr_status}


def _process_respiratory(state: SessionState, ts: float) -> dict:
    """Process respiratory rate from the same rPPG signal buffer."""
    brpm = None
    if len(state.raw_signals) >= 20:
        filtered = bandpass_filter_resp(np.array(state.raw_signals), state.fs)
        state.resp_signal = filtered.tolist()
        brpm = estimate_breathing_rate(np.array(state.resp_signal), state.fs)
        if brpm is not None:
            state.resp_history.append((ts, brpm))
    status = "ok" if brpm is not None else "insufficient_data"
    return {"brpm": round(brpm, 1) if brpm is not None else None, "status": status}


def _process_hrv_stress(state: SessionState, ts: float) -> dict:
    """Process HRV and relative stress score from pulse peaks."""
    ibis = compute_ibi(state.peak_times)
    rmssd = compute_rmssd(ibis)
    if rmssd is not None and state.baseline_rmssd is not None:
        state.rmssd_history.append((ts, rmssd))
    if rmssd is None:
        return {"status": "insufficient_beats",
                "label": "Not enough pulse peaks yet — keep still",
                "rmssd": None, "baseline_rmssd": state.baseline_rmssd}
    stress = rmssd_to_stress_score(rmssd, state.baseline_rmssd)
    stress["rmssd"] = round(rmssd, 2)
    return stress


def _process_drowsiness(state: SessionState,
                        landmarks: list[tuple[float, float]],
                        ts: float) -> tuple[dict, float]:
    """Process drowsiness from EAR and blink detection. Returns (result, blink_rate)."""
    left_ear = eye_aspect_ratio(landmarks, "left")
    right_ear = eye_aspect_ratio(landmarks, "right")
    ear_avg = (left_ear + right_ear) / 2.0
    state.ear_history.append(ear_avg)
    state.ear_timestamps.append(ts)
    blink_detected = detect_blink(state.ear_history)
    if blink_detected:
        state.blink_timestamps.append(ts)
    blink_rate = compute_blink_rate(state.blink_timestamps)
    perclos = compute_perclos(state.ear_history, state.ear_timestamps)
    drowsiness = assess_drowsiness(blink_rate, perclos)
    if drowsiness["status"] in ("drowsy", "mild_fatigue"):
        state.drowsiness_events.append((ts, drowsiness["status"]))
    return drowsiness, blink_rate


def _process_eye_strain(state: SessionState,
                        landmarks: list[tuple[float, float]],
                        blink_rate: float, ts: float) -> dict:
    """Process eye strain risk from distance, blink rate, and exposure."""
    distance = estimate_screen_distance(landmarks, state.calibration_interocular_px)
    exposure = screen_time_exposure(state.session_start)
    strain_risk = assess_eye_strain_risk(blink_rate, distance["status"], exposure)
    state.eye_strain_history.append((ts, strain_risk["risk"]))
    return strain_risk


def process_frame(state: SessionState, frame_data: dict) -> dict:
    """
    Process a single frame of landmark data through all four wellness modules.

    Args:
        state: Per-session state with rolling buffers.
        frame_data: Dict with landmarks, timestamp, roi_mean.

    Returns:
        dict with all four module outputs and frame status.
    """
    result: dict[str, Any] = {
        "frame_status": "ok",
        "timestamp": frame_data.get("timestamp", time.time()),
    }
    landmarks_raw = frame_data.get("landmarks")
    if not landmarks_raw or len(landmarks_raw) < 100:
        result["frame_status"] = "no_face"
        result["heart_rate"] = None
        result["respiratory_rate"] = None
        result["hrv"] = None
        result["drowsiness"] = None
        result["eye_strain"] = None
        state.no_face_count += 1
        return result

    landmarks = [(lm["x"], lm["y"]) for lm in landmarks_raw]
    roi_val = frame_data.get("roi_mean", 0.0)
    state.raw_signals.append(roi_val)
    ts = result["timestamp"]
    state.timestamps.append(ts)
    state.trim_buffers()
    _update_frame_rate(state)

    result["heart_rate"] = _process_heart_rate(state, ts)
    result["respiratory_rate"] = _process_respiratory(state, ts)
    result["hrv"] = _process_hrv_stress(state, ts)
    drowsiness, blink_rate = _process_drowsiness(state, landmarks, ts)
    result["drowsiness"] = drowsiness
    result["eye_strain"] = _process_eye_strain(state, landmarks, blink_rate, ts)
    result["frame_number"] = state.frame_count
    state.frame_count += 1
    return result


@app.websocket("/ws/session")
async def ws_session(ws: WebSocket) -> None:
    """WebSocket endpoint for live frame-by-frame wellness monitoring."""
    await ws.accept()
    session_id = str(id(ws))
    state = SessionState()
    sessions[session_id] = state

    try:
        while True:
            raw = await ws.receive_text()
            try:
                frame_data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            try:
                if frame_data.get("type") == "heartbeat":
                    continue

                if frame_data.get("type") == "calibration_start":
                    state.calibrating = True
                    state.calibration_start = time.time()
                    state.calibration_data = []
                    await ws.send_json({"type": "calibration_status",
                                        "status": "started",
                                        "duration_sec": CALIBRATION_DURATION_SEC})
                    continue

                if frame_data.get("type") == "calibration_data":
                    if state.calibrating:
                        state.calibration_data.append(frame_data)
                        elapsed = time.time() - state.calibration_start
                        if elapsed >= CALIBRATION_DURATION_SEC:
                            state.calibrating = False
                            await _finish_calibration(state)
                            await ws.send_json({"type": "calibration_complete",
                                                "baseline_rmssd": state.baseline_rmssd,
                                                "interocular_px": state.calibration_interocular_px})
                            continue
                    await ws.send_json({"type": "calibration_status",
                                        "status": "collecting",
                                        "elapsed": round(time.time() - state.calibration_start, 1)})
                    continue

                if frame_data.get("type") == "calibration_end":
                    state.calibrating = False
                    await _finish_calibration(state)
                    await ws.send_json({"type": "calibration_complete",
                                        "baseline_rmssd": state.baseline_rmssd,
                                        "interocular_px": state.calibration_interocular_px})
                    continue

                if frame_data.get("type") == "get_summary":
                    summary = _build_summary(state)
                    await ws.send_json({"type": "session_summary", "summary": summary})
                    save_session("wellness", time.time() - state.session_start, summary)
                    continue

                if frame_data.get("type") == "run_tests":
                    result = await _run_tests_async()
                    await ws.send_json({"type": "test_results", "data": result})
                    continue

                # ─── Neuro Exam Handlers ───
                msg_type = frame_data.get("type", "")
                
                if msg_type.startswith("neuro_"):
                    await _handle_neuro_message(ws, frame_data)
                    continue

                neuro_state = neuro_sessions.get(session_id)
                if not msg_type and neuro_state and neuro_state.current_test:
                    frame_data["type"] = "neuro_frame"
                    await _handle_neuro_message(ws, frame_data)
                    continue

                result = process_frame(state, frame_data)
                if state.calibrating:
                    if frame_data.get("landmarks"):
                        state.calibration_data.append(frame_data)
                    elapsed = time.time() - state.calibration_start
                    await ws.send_json({
                        "type": "calibration_status",
                        "status": "collecting",
                        "elapsed": round(elapsed, 1),
                        "duration_sec": CALIBRATION_DURATION_SEC,
                    })
                    if elapsed >= CALIBRATION_DURATION_SEC:
                        state.calibrating = False
                        await _finish_calibration(state)
                        await ws.send_json({
                            "type": "calibration_complete",
                            "baseline_rmssd": state.baseline_rmssd,
                            "interocular_px": state.calibration_interocular_px,
                        })
                await ws.send_json({"type": "frame_result", "data": result})
            except Exception as e:
                msg_type = frame_data.get("type", "") if "frame_data" in locals() else ""
                if not msg_type.startswith("neuro_") and msg_type != "heartbeat":
                    await ws.send_json({"type": "frame_result", "data": {
                        "frame_status": "error",
                        "heart_rate": None, "respiratory_rate": None,
                        "hrv": None, "drowsiness": None, "eye_strain": None,
                    }})


    except WebSocketDisconnect:
        pass
    finally:
        sessions.pop(session_id, None)


async def _run_tests_async() -> dict:
    """Run pytest asynchronously in a subprocess and return structured results."""
    import os
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=backend_dir,
            env={**os.environ, "PYTHONPATH": backend_dir},
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=30,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return {"status": "error", "error": "Test run timed out after 30 seconds"}

    lines = (stdout.decode() + stderr.decode()).split("\n")
    passed = 0
    failed = 0
    tests_detail = []
    for line in lines:
        if "PASSED" in line and "::" in line:
            passed += 1
        if "FAILED" in line and "::" in line:
            failed += 1
            name = line.split("::")[-1].split(" ")[0] if "::" in line else ""
            tests_detail.append({"name": name, "status": "failed"})
    return {
        "status": "ok" if proc.returncode == 0 else "failures",
        "passed": passed,
        "failed": failed,
        "total": passed + failed,
        "returncode": proc.returncode or 0,
        "output": (stdout.decode()[-2000:]),
    }


async def _finish_calibration(state: SessionState) -> None:
    """Compute baseline RMSSD and interocular distance from calibration data."""
    calib_signals: list[float] = []
    calib_landmarks_all: list[list[tuple[float, float]]] = []

    for entry in state.calibration_data:
        calib_signals.append(entry.get("roi_mean", 0.0))
        lms_raw = entry.get("landmarks")
        if lms_raw:
            calib_landmarks_all.append([(lm["x"], lm["y"]) for lm in lms_raw])

    # Compute interocular calibration distance
    if calib_landmarks_all:
        from config import LEFT_EYE_OUTER, RIGHT_EYE_OUTER
        dists = []
        for lms in calib_landmarks_all:
            if len(lms) > RIGHT_EYE_OUTER:
                p_l = np.array(lms[LEFT_EYE_OUTER])
                p_r = np.array(lms[RIGHT_EYE_OUTER])
                dists.append(float(np.linalg.norm(p_r - p_l)))
        if dists:
            state.calibration_interocular_px = float(np.mean(dists))

    # Compute baseline RMSSD from calibration signal
    if len(calib_signals) >= 5:
        calib_timestamps = [e.get("timestamp", 0) for e in state.calibration_data
                            if "timestamp" in e]
        if len(calib_timestamps) >= 2:
            intervals = np.diff(sorted(calib_timestamps))
            calib_fs = 1.0 / max(np.mean(intervals), 0.001)
        else:
            calib_fs = 30.0
        filtered = bandpass_filter(np.array(calib_signals), calib_fs)
        peaks = detect_pulse_peaks(filtered, calib_fs)
        peak_times = [peak / calib_fs for peak in peaks]
        ibis = compute_ibi(peak_times)
        rmssd_val = compute_rmssd(ibis)
        if rmssd_val is not None:
            state.baseline_rmssd = rmssd_val


def _compute_avg_resp(state: SessionState) -> float | None:
    """Compute average breathing rate from history."""
    if not state.resp_history:
        return None
    vals = [brpm for _, brpm in state.resp_history if brpm is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


def _build_summary(state: SessionState) -> dict:
    """Build end-of-session summary report."""
    avg_bpm = None
    if state.bpm_history:
        bpm_values = [bpm for _, bpm in state.bpm_history if bpm is not None]
        if bpm_values:
            avg_bpm = round(sum(bpm_values) / len(bpm_values), 1)
    rmssd_trend = None
    if state.rmssd_history:
        rmssd_trend = [(round(t, 1), round(r, 2)) for t, r in state.rmssd_history]
    total_blinks = len(state.blink_timestamps)
    drowsy_events = [(round(t, 1), s) for t, s in state.drowsiness_events]
    strain_timeline = [(round(t, 1), r) for t, r in state.eye_strain_history]
    total_frames = state.frame_count
    no_face_pct = round(state.no_face_count / max(total_frames, 1) * 100, 1)
    return {
        "session_duration_min": round((time.time() - state.session_start) / 60.0, 1),
        "average_heart_rate_bpm": avg_bpm,
        "average_breathing_rate_brpm": _compute_avg_resp(state),
        "rmssd_trend": rmssd_trend,
        "total_blinks": total_blinks,
        "drowsiness_events": drowsy_events,
        "eye_strain_timeline": strain_timeline,
        "total_frames_processed": total_frames,
        "no_face_percentage": no_face_pct,
        "disclaimer": "Wellness indicator, not a medical device. All metrics are "
                      "relative/qualitative estimates from signal processing.",
    }


async def _handle_neuro_message(ws: WebSocket, frame_data: dict) -> None:
    """Route neuro exam messages to appropriate test pipeline."""
    msg_type = frame_data.get("type", "")
    session_id = str(id(ws))
    state = neuro_sessions.get(session_id)

    if msg_type == "neuro_start":
        # Initialize a new neuro session
        state = NeuroSessionState()
        neuro_sessions[session_id] = state
        test = frame_data.get("test", "")
        state.current_test = test
        state.test_phase = "instructions"
        state.test_side = frame_data.get("side", "right")
        await ws.send_json({"type": "neuro_status", "status": "started",
                            "test": test, "phase": "instructions"})
        return

    if not state:
        await ws.send_json({"type": "neuro_error", "error": "No active neuro session"})
        return

    if msg_type == "neuro_frame":
        test = state.current_test
        if test == "finger_nose":
            _handle_finger_nose_frame(state, frame_data)
        elif test == "dysdiadochokinesia":
            _handle_ddk_frame(state, frame_data)
        elif test == "romberg":
            _handle_romberg_frame(state, frame_data)

    elif msg_type == "neuro_phase_change":
        phase = frame_data.get("phase", "")
        state.test_phase = phase
        state.phase_start = time.time()
        if state.current_test == "romberg":
            if phase == "eyes_open":
                state.eyes_open_com = []
            elif phase == "eyes_closed":
                state.eyes_closed_com = []
            elif phase == "results":
                romberg = assess_romberg(state.eyes_open_com, state.eyes_closed_com)
                state.romberg_results = romberg
                await ws.send_json({"type": "neuro_result",
                                    "test": "romberg", "data": romberg})
                return
        await ws.send_json({"type": "neuro_phase", "phase": phase,
                            "test": state.current_test})

    elif msg_type == "neuro_complete":
        report = _build_neuro_report(state)
        await ws.send_json({"type": "neuro_report", "data": report})
        save_session("neuro_exam", time.time() - state.session_start, report)
        neuro_sessions.pop(session_id, None)

    elif msg_type == "neuro_cancel":
        if state.current_test:
            if state.current_test == "finger_nose":
                if state.trajectories:
                    trial = {
                        "side": state.test_side,
                        "trajectory": [(e["x"], e["y"], e["ts"]) for e in state.trajectories],
                        "nose_pos": (state.trajectories[0]["nose_x"], state.trajectories[0]["nose_y"]),
                        "touch_frame_idx": -1,
                    }
                    state.finger_nose_results = assess_finger_to_nose([trial])
                await ws.send_json({"type": "neuro_result",
                                    "test": "finger_nose",
                                    "data": state.finger_nose_results})
            elif state.current_test == "dysdiadochokinesia":
                await ws.send_json({"type": "neuro_result",
                                    "test": "dysdiadochokinesia",
                                    "data": state.ddk_results})
        neuro_sessions.pop(session_id, None)
        await ws.send_json({"type": "neuro_status", "status": "cancelled"})


def _handle_finger_nose_frame(state: NeuroSessionState, data: dict) -> None:
    """Accumulate finger-to-nose trajectory data."""
    side = state.test_side
    
    pose = data.get("pose_landmarks")
    if not pose:
        return
        
    nose = {"x": pose[0]["x"], "y": pose[0]["y"]} if len(pose) > 0 else {}
    
    hand = data.get(f"{side}_hand")
    if hand and len(hand) > 8:
        point = {"x": hand[8]["x"], "y": hand[8]["y"]}
    else:
        # Fallback to wrist
        idx = 15 if side == "left" else 16
        point = {"x": pose[idx]["x"], "y": pose[idx]["y"]} if len(pose) > idx else {}
        
    ts = data.get("timestamp", time.time())
    phase = state.test_phase
    
    entry = {
        "side": side,
        "x": point.get("x", 0),
        "y": point.get("y", 0),
        "ts": ts,
        "nose_x": nose.get("x", 0),
        "nose_y": nose.get("y", 0),
        "phase": phase,
    }
    state.trajectories.append(entry)


def _handle_ddk_frame(state: NeuroSessionState, data: dict) -> None:
    """Accumulate dysdiadochokinesia palm angle data."""
    side = state.test_side
    hand_lms = data.get(f"{side}_hand", [])
    if not hand_lms:
        return
    ts = data.get("timestamp", time.time())
    angle = compute_palm_orientation(hand_lms)
    if angle is not None:
        state.palm_angle_buffer.append(angle)
        state.angle_timestamps.append(ts)
    if state.test_phase == "complete" or data.get("phase") == "complete":
        if len(state.angle_timestamps) >= 2:
            fs = 1.0 / max(np.mean(np.diff(state.angle_timestamps[-30:])), 0.001)
        else:
            fs = 30.0
        state.ddk_results[side] = assess_dysdiadochokinesia(
            state.palm_angle_buffer, fs, side)


def _handle_romberg_frame(state: NeuroSessionState, data: dict) -> None:
    """Accumulate center-of-mass data for Romberg test."""
    pose_lms = data.get("pose_landmarks", [])
    com = extract_center_of_mass(pose_lms)
    if com is None:
        return
    ts = data.get("timestamp", time.time())
    if state.test_phase == "eyes_open":
        state.eyes_open_com.append(com)
    elif state.test_phase == "eyes_closed":
        state.eyes_closed_com.append(com)


def _build_neuro_report(state: NeuroSessionState) -> dict:
    """Compile final neuro exam report from all test results."""
    return {
        "exam_duration_s": round(time.time() - state.session_start, 1),
        "tests_completed": [
            t for t, r in [
                ("finger_to_nose", state.finger_nose_results),
                ("dysdiadochokinesia", state.ddk_results),
                ("romberg", state.romberg_results),
            ] if r
        ],
        "finger_to_nose": state.finger_nose_results or None,
        "dysdiadochokinesia": state.ddk_results or None,
        "romberg": state.romberg_results or None,
        "disclaimer": (
            "Quantitative motor screen using computer vision. "
            "Not a clinical diagnostic device. Results are derived from "
            "geometric and signal processing computations on webcam landmarks."
        ),
    }


# ─── History REST Endpoints ───

@app.get("/api/history")
async def get_history(
    session_type: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
):
    """List recent session history from the database."""
    return list_sessions(limit=limit, session_type=session_type)


@app.get("/api/history/{session_id}")
async def get_history_session(session_id: int):
    """Retrieve a single session by ID."""
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.delete("/api/history/{session_id}")
async def delete_history_session(session_id: int):
    """Delete a session record by ID."""
    deleted = delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "ok", "deleted": True}


@app.post("/api/calibrate")
async def calibrate():
    """
    Placeholder for REST-based calibration trigger.
    Production calibration happens via the WebSocket calibration flow.
    """
    return {"status": "use WebSocket session for calibration flow"}


@app.get("/api/status")
async def api_status():
    """
    Return backend health status including uptime and active sessions.
    """
    uptime_sec = time.time() - _startup_time
    return {
        "status": "online",
        "version": "0.1.0",
        "uptime_seconds": round(uptime_sec, 1),
        "active_sessions": len(sessions),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/api/tests/run")
async def run_tests():
    """
    Run pytest programmatically and return structured results.
    Returns pass/fail counts, individual test results, and full output.
    """
    import os
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
            capture_output=True, text=True, timeout=30,
            cwd=backend_dir,
            env={**os.environ, "PYTHONPATH": backend_dir},
        )
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "Test run timed out after 30 seconds"}
    except FileNotFoundError:
        return {"status": "error", "error": "pytest not found — install pytest first"}

    lines = result.stdout.split("\n") + result.stderr.split("\n")
    passed = 0
    failed = 0
    skipped = 0
    tests_detail = []
    for line in lines:
        if "PASSED" in line and ("::" in line):
            passed += 1
            name = line.split("::")[-1].split(" ")[0] if "::" in line else ""
            tests_detail.append({"name": name, "status": "passed"})
        if "FAILED" in line and ("::" in line):
            failed += 1
            name = line.split("::")[-1].split(" ")[0] if "::" in line else ""
            tests_detail.append({"name": name, "status": "failed"})
    skipped = result.stdout.count("SKIPPED")

    return {
        "status": "ok" if result.returncode == 0 else "failures",
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "total": passed + failed + skipped,
        "returncode": result.returncode,
        "output": result.stdout[-2000:],
        "tests": tests_detail[-50:],
    }


@app.get("/api/session/summary")
async def session_summary():
    """
    Placeholder for REST-based summary retrieval.
    Production summary is obtained via WebSocket 'get_summary' message.
    """
    return {"status": "use WebSocket session for summary retrieval"}


@app.post("/api/report/wellness")
async def export_wellness_report(summary: dict):
    """
    Generate a PDF report from a wellness session summary.
    Accepts session summary JSON in the request body.
    """
    try:
        pdf = generate_wellness_report(summary)
        return Response(
            content=pdf.read(),
            media_type="application/pdf",
            headers={"Content-Disposition": "inline; filename=wellness_report.pdf"},
        )
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/api/report/neuro")
async def export_neuro_report(summary: dict):
    """
    Generate a PDF report from a neuro exam session summary.
    Accepts exam summary JSON in the request body.
    """
    try:
        pdf = generate_neuro_report(summary)
        return Response(
            content=pdf.read(),
            media_type="application/pdf",
            headers={"Content-Disposition": "inline; filename=neuro_report.pdf"},
        )
    except Exception as e:
        return {"status": "error", "error": str(e)}

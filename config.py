"""
All tunable constants for the wellness monitor.
No magic numbers elsewhere — every threshold, window, and range lives here.
"""

# --- Signal Buffer ---
SIGNAL_BUFFER_SECONDS = 30           # rolling buffer duration for landmark/signal data
TARGET_FPS = 30                       # target webcam framerate (never assumed for math)

# --- Heart Rate (rPPG) ---
BANDPASS_LOW_HZ = 0.7                 # 42 BPM lower bound
BANDPASS_HIGH_HZ = 4.0                # 240 BPM upper bound
BANDPASS_ORDER = 4                    # Butterworth filter order
MIN_SIGNAL_SECONDS = 10.0             # minimum buffered signal before estimating BPM
MAX_HEART_RATE_BPM = 240.0            # physiologically plausible ceiling
PEAK_MIN_DISTANCE_SEC = 60.0 / MAX_HEART_RATE_BPM  # minimum seconds between peaks

# --- Respiratory Rate (from rPPG signal) ---
RESP_BANDPASS_LOW_HZ = 0.1            # 6 breaths/min lower bound
RESP_BANDPASS_HIGH_HZ = 0.5           # 30 breaths/min upper bound
MIN_SIGNAL_SECONDS_RESP = 15.0        # minimum buffered signal for respiratory estimate

# --- POS Algorithm ---
POS_TEMPORAL_WINDOW = 1.6             # seconds for temporal normalization in POS
POS_EPS = 1e-7                        # epsilon for division safety

# --- MediaPipe Face Mesh Landmark Indices ---
# Forehead ROI for rPPG: a bounding region on the forehead spanning key landmarks
FOREHEAD_LANDMARKS = [10, 67, 69, 104, 108, 109, 151, 299, 337, 338]

# Eye landmarks (MediaPipe Face Mesh indices for left and right eyes)
# Using 6 landmark points per eye for EAR calculation
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]    # left eye contour
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]  # right eye contour

# Interocular distance landmarks (outer corners of left/right eyes for calibration)
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

# --- Drowsiness ---
EAR_THRESHOLD = 0.21                  # eye aspect ratio threshold for closed eye
CONSECUTIVE_FRAMES_EYE_CLOSE = 2      # frames below threshold to count as blink state
BLINK_RATE_WINDOW_SEC = 60.0          # trailing window for blink rate (blinks/min)
PERCLOS_WINDOW_SEC = 60.0             # trailing window for PERCLOS calculation

# Drowsiness assessment thresholds (heuristic, not clinically validated)
BLINK_RATE_LOW = 10.0                 # blinks/min — low end of alert range
BLINK_RATE_DROWSY = 20.0              # elevated blink rate may indicate drowsiness
PERCLOS_MILD_THRESHOLD = 0.15         # 15% of time eyes closed = mild fatigue
PERCLOS_DROWSY_THRESHOLD = 0.30       # 30% = drowsy

# --- Eye Strain ---
EYE_STRAIN_LOW_BLINK_THRESHOLD = 8.0  # blinks/min below this contributes to strain risk
EXPOSURE_LOW_THRESHOLD = 30.0         # minutes — low exposure threshold
EXPOSURE_HIGH_THRESHOLD = 60.0        # minutes — high exposure threshold
DISTANCE_FAR_THRESHOLD = 0.85      # ratio below this = farther than calibration
DISTANCE_NEAR_THRESHOLD = 1.15     # ratio above this = closer than calibration

# --- Calibration ---
CALIBRATION_DURATION_SEC = 90.0       # baseline calibration period in seconds

# --- API ---
CORS_ORIGINS = ["http://localhost:5173", "http://localhost:3000", "https://wellness-frontend-xi.vercel.app", "*"]
WS_HEARTBEAT_SEC = 5.0                # WebSocket keepalive interval
WS_MAX_FRAME_AGE_SEC = 2.0            # discard frames older than this when processing

# --- Neuro Exam ---
# Finger-to-Nose test
TREMOR_FREQ_LOW_HZ = 3.0              # tremor frequency range lower bound
TREMOR_FREQ_HIGH_HZ = 12.0            # tremor frequency range upper bound
TREMOR_AMPLITUDE_THRESHOLD_PX = 4.0   # RMS tremor amplitude above this = significant (Fahn-Tolosa-Marin)
DYSMETRIA_THRESHOLD_PX = 30.0         # distance to nose above this = overshoot
MOVEMENT_TIME_ASYMMETRY_THRESHOLD = 0.3  # 30% time difference between sides flagged

# Dysdiadochokinesia (rapid alternating hand movements)
DDK_RATE_LOW_HZ = 0.5                 # lower bound for movement rate detection
DDK_RATE_HIGH_HZ = 8.0                # upper bound
DDK_RHYTHM_CV_THRESHOLD = 0.3         # CV above this = arrhythmic
DDK_AMPLITUDE_DECAY_THRESHOLD = 30.0  # % decay above this = significant fatigue/decline

# Romberg test
ROMBER_QUOTIENT_NORMAL_MAX = 2.0      # Romberg quotient below this = normal (Lanska & Goetz 2000)
ROMBER_QUOTIENT_MILD_MAX = 3.0        # 2.0-3.0 = mild impairment
EYES_OPEN_DURATION_SEC = 30.0         # eyes-open phase duration
EYES_CLOSED_DURATION_SEC = 30.0       # eyes-closed phase duration

# MediaPipe Pose landmark indices (33 total)
POSE_NOSE = 0
POSE_LSHOULDER = 11
POSE_RSHOULDER = 12
POSE_LHIP = 23
POSE_RHIP = 24
POSE_LKNEE = 25
POSE_RKNEE = 26
POSE_LANKLE = 27
POSE_RANKLE = 28
POSE_LWRIST = 15
POSE_RWRIST = 16
POSE_LELBOW = 13
POSE_RELBOW = 14

# MediaPipe Hands landmark indices (21 total)
HAND_WRIST = 0
HAND_INDEX_TIP = 8
HAND_INDEX_MCP = 5
HAND_PINKY_MCP = 17
HAND_THUMB_TIP = 4

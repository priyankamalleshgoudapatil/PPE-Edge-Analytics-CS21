"""
=============================================================================
CS21: PPE (Helmet/Vest) Detection on Site - Edge Analytics
Configuration Module (config.py)
-----------------------------------------------------------------------------
All tunable hyperparameters, thresholds, file paths, and class labels
are centralized in this file. Designed for easy tuning and viva demonstration.
=============================================================================
"""

import os
from pathlib import Path

# --- Base Directory Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VIDEOS_DIR = DATA_DIR / "videos"
SAMPLES_DIR = DATA_DIR / "samples"
MODELS_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"
RESULTS_DIR = BASE_DIR / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"
LOGS_DIR = RESULTS_DIR / "alert_logs"
SNAPSHOTS_DIR = RESULTS_DIR / "snapshots"
SCREENSHOTS_DIR = RESULTS_DIR / "screenshots"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Test Video Filepaths (Step 2 Scenarios)
VIDEO_NORMAL_PATH = VIDEOS_DIR / "normal_scenario.mp4"
VIDEO_ABNORMAL_PATH = VIDEOS_DIR / "abnormal_scenario.mp4"
VIDEO_CRITICAL_PATH = VIDEOS_DIR / "critical_scenario.mp4"

# Video generation & stream parameters
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_FPS = 30
VIDEO_FRAME_COUNT = 150  # 5 seconds duration at 30 fps

# Ensure all output directories exist
for folder in [DATA_DIR, VIDEOS_DIR, SAMPLES_DIR, MODELS_DIR, RESULTS_DIR, GRAPHS_DIR, LOGS_DIR, SNAPSHOTS_DIR, SCREENSHOTS_DIR, NOTEBOOKS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# --- Model & LiteRT Inference Settings ---
# YOLOv8n exported to LiteRT/TFLite
MODEL_FILENAME = "ppe_model.tflite"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME

# YOLOv8 default input dimensions
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
INPUT_CHANNELS = 3

# Preprocessing Settings (Step 3)
ENABLE_BRIGHTNESS_CORRECTION = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
ENABLE_BLUR_DETECTION = True
BLUR_LAPLACIAN_THRESHOLD = 50.0  # Variance below this threshold indicates motion blur

# Inference Thresholds (Step 4)
CONFIDENCE_THRESHOLD = 0.50   # Minimum box detection confidence
IOU_THRESHOLD = 0.45          # Non-Maximum Suppression (NMS) threshold

# Model Class Labels (5 classes)
CLASS_NAMES = [
    "person",      # Class 0: Detected worker/person
    "helmet",      # Class 1: Wearing helmet/hard-hat (Safety compliant)
    "no_helmet",   # Class 2: Missing helmet (Violation)
    "vest",        # Class 3: Wearing reflective vest (Safety compliant)
    "no_vest"      # Class 4: Missing safety vest (Violation)
]

# Color map for bounding box visualization (BGR format for OpenCV)
CLASS_COLORS = {
    "person": (255, 255, 255),    # White
    "helmet": (0, 255, 0),        # Green (Compliant)
    "no_helmet": (0, 0, 255),     # Red (Non-compliant)
    "vest": (0, 255, 0),          # Green (Compliant)
    "no_vest": (0, 0, 255)        # Red (Non-compliant)
}

# --- Edge Decision Logic Rules (Step 5) ---
# Consecutive frame thresholds to eliminate single-frame transient sensor noise
WARNING_CONSECUTIVE_FRAMES = 5    # Missing 1 item for >= 5 consecutive frames
CRITICAL_CONSECUTIVE_FRAMES = 15  # Missing both OR single violation >= 15 consecutive frames

# System Safety Statuses
STATUS_NORMAL = "NORMAL"
STATUS_WARNING = "WARNING"
STATUS_CRITICAL = "CRITICAL"

STATUS_COLORS = {
    STATUS_NORMAL: (0, 255, 0),       # Green
    STATUS_WARNING: (0, 191, 255),    # Amber / Orange (BGR: 0, 191, 255)
    STATUS_CRITICAL: (0, 0, 255)      # Red
}

# Prescriptive Action Prescriptions (Edge Prescriptive Analytics)
PRESCRIPTIVE_ACTIONS = {
    STATUS_NORMAL: "PPE COMPLIANT: All site safety regulations met. Continue regular operations.",
    STATUS_WARNING: "CAUTION: Partial PPE violation detected. Issue audio warning & verify compliance.",
    STATUS_CRITICAL: "HAZARD: Severe PPE non-compliance! Halt heavy machinery and alert site safety manager."
}

# --- Edge-to-Cloud Logging & Networking (Step 7) ---
DB_NAME = "ppe_alerts.db"
DB_PATH = LOGS_DIR / DB_NAME
CSV_PATH = LOGS_DIR / "ppe_alerts.csv"
SUMMARY_CSV_PATH = LOGS_DIR / "ppe_1min_summary.csv"

# Cloud connectivity toggle
CLOUD_CONNECTED_DEFAULT = True

# Cloud network latency simulation bounds (in milliseconds)
SIMULATED_CLOUD_LATENCY_MIN_MS = 200.0   # e.g., 4G/LTE round trip + queueing
SIMULATED_CLOUD_LATENCY_MAX_MS = 2000.0  # high congestion or weak site signal

# Summary reporting interval
SUMMARY_INTERVAL_SECONDS = 60.0

# --- General System Settings ---
RANDOM_SEED = 42
DEBUG_MODE = True

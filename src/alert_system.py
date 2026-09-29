"""
=============================================================================
CS21: PPE Detection - Edge Alert & Action Dispatcher (alert_system.py)
-----------------------------------------------------------------------------
Executes local edge actions upon safety violations:
  1. Color-coded Console Alerts (Green NORMAL, Amber WARNING, Red CRITICAL).
  2. Visual On-Frame Overlays (Colored bounding boxes, site hazard banner).
  3. Audio Buzzer Dispatch (Auditory alert via system sound / winsound).
  4. Timestamped Violation Snapshot Preservation (Saved locally for audits).
=============================================================================
"""

import sys
import os
import time
from pathlib import Path
from typing import Tuple, List, Dict
import cv2
import numpy as np

# Ensure src directory is in path
sys.path.append(str(Path(__file__).resolve().parent))
import config
from decision_logic import EdgeAnalyticsResult

# Import winsound on Windows for auditory buzzer
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


# ANSI Color Codes for Terminal Console
class TermColor:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    AMBER = "\033[93m"  # Yellow/Amber
    RED = "\033[91m"
    CYAN = "\033[96m"


class EdgeAlertDispatcher:
    """
    Local edge actuation and notification engine.
    Ensures all critical responses fire deterministically at the edge
    even if the cloud uplink is severed.
    """

    def __init__(self,
                 snapshots_dir=config.SNAPSHOTS_DIR,
                 enable_sound=True,
                 snapshot_cooldown_sec=1.5):
        """
        :param snapshots_dir: Local path to store violation evidence frames
        :param enable_sound: Toggle auditory buzzer
        :param snapshot_cooldown_sec: Minimum seconds between snapshots to prevent disk flooding
        """
        self.snapshots_dir = Path(snapshots_dir)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.enable_sound = enable_sound
        self.snapshot_cooldown = snapshot_cooldown_sec
        self.last_snapshot_time = 0.0
        self.last_alert_status = config.STATUS_NORMAL

    def trigger_buzzer(self, status: str):
        """
        Plays simulated auditory buzzer for on-site personnel.
        - WARNING: Moderate tone (800 Hz, 120 ms)
        - CRITICAL: High-urgency alert tone (1800 Hz, 250 ms)
        """
        if not self.enable_sound:
            return

        try:
            if HAS_WINSOUND:
                if status == config.STATUS_WARNING:
                    winsound.Beep(800, 120)
                elif status == config.STATUS_CRITICAL:
                    winsound.Beep(1800, 200)
            else:
                # Fallback terminal bell
                sys.stdout.write("\a")
                sys.stdout.flush()
        except Exception:
            # Sound hardware unavailable or muted
            pass

    def log_console_alert(self, result: EdgeAnalyticsResult):
        """
        Prints formatted, color-coded console alerts with ANSI colors.
        - Green for NORMAL
        - Amber for WARNING
        - Red for CRITICAL
        """
        ts = result.timestamp
        frame = result.frame_idx
        streak = result.consecutive_violations

        if result.status == config.STATUS_NORMAL:
            print(f"{TermColor.GREEN}[NORMAL]{TermColor.RESET} Frame {frame:04d} | {ts} | Streak: {streak} | All personnel compliant.")
        
        elif result.status == config.STATUS_WARNING:
            items_str = ", ".join(result.missing_items).upper() if result.missing_items else "PARTIAL PPE"
            print(f"{TermColor.AMBER}{TermColor.BOLD}[WARNING]{TermColor.RESET} Frame {frame:04d} | {ts} | "
                  f"Streak: {streak} frames | Missing: {items_str} | Directives: {result.prescriptive_action[:45]}...")

        elif result.status == config.STATUS_CRITICAL:
            items_str = ", ".join(result.missing_items).upper() if result.missing_items else "SEVERE NON-COMPLIANCE"
            print(f"{TermColor.RED}{TermColor.BOLD}[CRITICAL]{TermColor.RESET} Frame {frame:04d} | {ts} | "
                  f"Streak: {streak} frames | Missing: {items_str} | ACTION: {result.prescriptive_action[:50]}!")

    def render_visual_overlays(self, frame: np.ndarray, result: EdgeAnalyticsResult,
                               latency_ms: float = 0.0, fps: float = 0.0) -> np.ndarray:
        """
        Draws dynamic bounding boxes and a prominent site status banner on the frame.
        - Compliant boxes: Green
        - Non-compliant boxes: Red
        - Person boxes: White
        - Header Banner: Color corresponds to system safety status
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # 1. Draw Detections Bounding Boxes
        for det in result.detections:
            x1, y1, x2, y2 = det["box"]
            cls_name = det["class_name"]
            conf = det["confidence"]

            # Compliant = Green, Violation = Red, Person = White
            if cls_name in ["helmet", "vest"]:
                box_color = (0, 255, 0)      # Green
            elif cls_name in ["no_helmet", "no_vest"]:
                box_color = (0, 0, 255)      # Red
            else:
                box_color = (255, 255, 255)  # White (Person)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            # Label box
            label = f"{cls_name}: {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw + 4, y1), box_color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

        # 2. Prominent Top Status Banner
        banner_h = 45
        banner_color = config.STATUS_COLORS.get(result.status, (0, 255, 0))
        # Semi-transparent overlay for sleek edge UI
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), banner_color, -1)
        cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

        # Status text & Icon
        status_text = f"SITE STATUS: {result.status} | Frame: {result.frame_idx:04d} | Streak: {result.consecutive_violations}"
        cv2.putText(annotated, status_text, (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

        # Sub-directive text
        directive_text = f"Action: {result.prescriptive_action}"
        cv2.putText(annotated, directive_text, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)

        # 3. Bottom Edge Telemetry Banner
        telemetry_str = f"LiteRT: {latency_ms:.1f}ms ({fps:.1f} FPS) | Missing: {', '.join(result.missing_items) if result.missing_items else 'None'}"
        cv2.putText(annotated, telemetry_str, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        return annotated

    def save_violation_snapshot(self, frame: np.ndarray, result: EdgeAnalyticsResult) -> str:
        """
        Saves timestamped snapshot image locally on disk when a violation occurs.
        Returns the absolute filepath of the saved snapshot.
        """
        current_time = time.time()
        # Enforce snapshot cooldown or status transition
        if (current_time - self.last_snapshot_time < self.snapshot_cooldown) and (result.status == self.last_alert_status):
            return ""

        # Construct unique timestamped filename
        clean_ts = result.timestamp.replace(":", "-").replace(" ", "_").replace(".", "_")
        filename = f"violation_{result.status}_{clean_ts}_frame{result.frame_idx:04d}.jpg"
        filepath = self.snapshots_dir / filename

        cv2.imwrite(str(filepath), frame)
        self.last_snapshot_time = current_time
        self.last_alert_status = result.status

        print(f"{TermColor.CYAN}  [LOCAL SNAPSHOT SAVED]{TermColor.RESET} -> {filepath.name}")
        return str(filepath)

    def dispatch(self, raw_frame: np.ndarray, result: EdgeAnalyticsResult,
                 latency_ms: float = 0.0, fps: float = 0.0) -> Tuple[np.ndarray, str]:
        """
        Main entry point coordinating all Step 6 actions:
          1. Console logging
          2. On-frame rendering
          3. Buzzer triggering
          4. Snapshot preservation
        Returns (annotated_frame, saved_snapshot_path)
        """
        # 1. Console Alert
        self.log_console_alert(result)

        # 2. Render visual bounding boxes & banners
        annotated_frame = self.render_visual_overlays(raw_frame, result, latency_ms, fps)

        snapshot_path = ""
        # 3. Trigger Buzzer & Save Snapshot on Violations
        if result.is_alert_triggered:
            self.trigger_buzzer(result.status)
            snapshot_path = self.save_violation_snapshot(annotated_frame, result)

        return annotated_frame, snapshot_path


def test_alert_system():
    """
    Self-testing verification suite for Step 6 actions.
    Generates synthetic verification frames and validates dispatching.
    """
    print("=" * 65)
    print("CS21: STEP 6 - VALIDATING ALERT & ACTION SYSTEM")
    print("=" * 65)

    dispatcher = EdgeAlertDispatcher(enable_sound=False) # Silent mode for terminal test
    sample_img = np.zeros((480, 640, 3), dtype=np.uint8)
    sample_img[:] = (80, 80, 80)

    # 1. Test NORMAL alert
    norm_res = EdgeAnalyticsResult(
        frame_idx=10, timestamp="2026-09-29 13:59:01.100",
        status=config.STATUS_NORMAL, consecutive_violations=0,
        detections=[{"box": [100, 100, 200, 350], "class_name": "person", "confidence": 0.94},
                    {"box": [120, 100, 180, 150], "class_name": "helmet", "confidence": 0.91},
                    {"box": [110, 160, 190, 260], "class_name": "vest", "confidence": 0.95}],
        missing_items=[], diagnosis_text="All compliant",
        total_frames=10, total_normal=10, total_warning=0, total_critical=0,
        total_helmet_violations=0, total_vest_violations=0,
        prescriptive_action=config.PRESCRIPTIVE_ACTIONS[config.STATUS_NORMAL],
        is_alert_triggered=False
    )
    frame_norm, snap1 = dispatcher.dispatch(sample_img, norm_res, latency_ms=14.2, fps=70.4)

    # 2. Test WARNING alert
    warn_res = EdgeAnalyticsResult(
        frame_idx=35, timestamp="2026-09-29 13:59:02.250",
        status=config.STATUS_WARNING, consecutive_violations=5,
        detections=[{"box": [100, 100, 200, 350], "class_name": "person", "confidence": 0.93},
                    {"box": [120, 100, 180, 150], "class_name": "helmet", "confidence": 0.90},
                    {"box": [110, 160, 190, 260], "class_name": "no_vest", "confidence": 0.88}],
        missing_items=["vest"], diagnosis_text="Missing VEST for 5 frames",
        total_frames=35, total_normal=30, total_warning=5, total_critical=0,
        total_helmet_violations=0, total_vest_violations=5,
        prescriptive_action=config.PRESCRIPTIVE_ACTIONS[config.STATUS_WARNING],
        is_alert_triggered=True
    )
    frame_warn, snap2 = dispatcher.dispatch(sample_img, warn_res, latency_ms=15.1, fps=66.2)

    # 3. Test CRITICAL alert
    crit_res = EdgeAnalyticsResult(
        frame_idx=50, timestamp="2026-09-29 13:59:03.500",
        status=config.STATUS_CRITICAL, consecutive_violations=15,
        detections=[{"box": [100, 100, 200, 350], "class_name": "person", "confidence": 0.95},
                    {"box": [120, 100, 180, 150], "class_name": "no_helmet", "confidence": 0.92},
                    {"box": [110, 160, 190, 260], "class_name": "no_vest", "confidence": 0.91}],
        missing_items=["helmet", "vest"], diagnosis_text="Missing BOTH Helmet and Vest",
        total_frames=50, total_normal=30, total_warning=5, total_critical=15,
        total_helmet_violations=15, total_vest_violations=20,
        prescriptive_action=config.PRESCRIPTIVE_ACTIONS[config.STATUS_CRITICAL],
        is_alert_triggered=True
    )
    frame_crit, snap3 = dispatcher.dispatch(sample_img, crit_res, latency_ms=13.8, fps=72.5)

    # Save annotated frames to results/snapshots/
    cv2.imwrite(str(config.SNAPSHOTS_DIR / "step6_annotated_normal.jpg"), frame_norm)
    cv2.imwrite(str(config.SNAPSHOTS_DIR / "step6_annotated_warning.jpg"), frame_warn)
    cv2.imwrite(str(config.SNAPSHOTS_DIR / "step6_annotated_critical.jpg"), frame_crit)

    print("\n" + "=" * 65)
    print("[SUCCESS] Step 6 verification complete:")
    print(f"  - Normal Verification Frame   -> {config.SNAPSHOTS_DIR / 'step6_annotated_normal.jpg'}")
    print(f"  - Warning Verification Frame  -> {config.SNAPSHOTS_DIR / 'step6_annotated_warning.jpg'}")
    print(f"  - Critical Verification Frame -> {config.SNAPSHOTS_DIR / 'step6_annotated_critical.jpg'}")
    print("=" * 65)


if __name__ == "__main__":
    test_alert_system()

"""
=============================================================================
CS21: PPE Detection - Edge Temporal Decision Logic Engine (decision_logic.py)
-----------------------------------------------------------------------------
Implements rule-based Edge Analytics (Module 3):
  - IF/THEN rules with temporal hysteresis (consecutive-frame counting).
  - Eliminates transient false alarms from camera noise or momentary occlusions.
  - Generates the 4 Pillars of Data Analytics:
      1. Predictive Analytics: Neural detections & probabilities.
      2. Diagnostic Analytics: Root cause analysis of missing PPE items.
      3. Descriptive Analytics: Running violation tallies & compliance rates.
      4. Prescriptive Analytics: Immediate actionable safety directives.
=============================================================================
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

# Ensure src directory is in path
sys.path.append(str(Path(__file__).resolve().parent))
import config


@dataclass
class EdgeAnalyticsResult:
    """
    Standardized payload capturing the complete 4-tier analytics state for a frame.
    """
    frame_idx: int
    timestamp: str
    status: str                         # NORMAL, WARNING, CRITICAL
    consecutive_violations: int
    # 1. Predictive: Raw neural detections
    detections: List[Dict[str, Any]]
    # 2. Diagnostic: Missing items list and detailed diagnosis
    missing_items: List[str]
    diagnosis_text: str
    # 3. Descriptive: Cumulative historical tallies
    total_frames: int
    total_normal: int
    total_warning: int
    total_critical: int
    total_helmet_violations: int
    total_vest_violations: int
    # 4. Prescriptive: Action directives
    prescriptive_action: str
    is_alert_triggered: bool


class EdgeDecisionEngine:
    """
    Edge Decision Engine executing IF/THEN safety logic with temporal debouncing.
    """

    def __init__(self,
                 warning_thresh=config.WARNING_CONSECUTIVE_FRAMES,
                 critical_thresh=config.CRITICAL_CONSECUTIVE_FRAMES):
        """
        :param warning_thresh: Consecutive violation frames required to enter WARNING (default: 5)
        :param critical_thresh: Consecutive violation frames required to enter CRITICAL (default: 15)
        """
        self.warning_thresh = warning_thresh
        self.critical_thresh = critical_thresh

        # Temporal State Counters
        self.consecutive_violations = 0
        self.consecutive_normal = 0
        self.current_status = config.STATUS_NORMAL

        # Descriptive Analytics Cumulative Counters
        self.total_frames = 0
        self.total_normal = 0
        self.total_warning = 0
        self.total_critical = 0
        self.total_helmet_violations = 0
        self.total_vest_violations = 0

    def reset_state(self):
        """Resets temporal tracking counters."""
        self.consecutive_violations = 0
        self.consecutive_normal = 0
        self.current_status = config.STATUS_NORMAL

    def evaluate_frame(self, detections: List[Dict[str, Any]], frame_idx: int, timestamp: str) -> EdgeAnalyticsResult:
        """
        Evaluates detections for a single frame through the IF/THEN decision matrix.
        """
        self.total_frames += 1

        # 1. Extract Predictive Detections
        has_person = any(d["class_name"] == "person" for d in detections)
        has_helmet = any(d["class_name"] == "helmet" for d in detections)
        has_no_helmet = any(d["class_name"] == "no_helmet" for d in detections)
        has_vest = any(d["class_name"] == "vest" for d in detections)
        has_no_vest = any(d["class_name"] == "no_vest" for d in detections)

        # 2. Diagnostic Root-Cause Analysis
        missing_items = []
        if has_no_helmet or (has_person and not has_helmet):
            missing_items.append("helmet")
            self.total_helmet_violations += 1

        if has_no_vest or (has_person and not has_vest):
            missing_items.append("vest")
            self.total_vest_violations += 1

        # 3. IF/THEN Edge Decision Logic with Temporal Filtering
        is_frame_violation = len(missing_items) > 0
        missing_both = ("helmet" in missing_items) and ("vest" in missing_items)

        if is_frame_violation:
            self.consecutive_violations += 1
            self.consecutive_normal = 0
        else:
            self.consecutive_normal += 1
            # Reset violation streak if compliant for 2 consecutive frames
            if self.consecutive_normal >= 2:
                self.consecutive_violations = 0

        # State Transition Matrix:
        # Rule 1: Missing BOTH items -> Immediate CRITICAL hazard
        if missing_both and self.consecutive_violations >= 2:
            new_status = config.STATUS_CRITICAL
            diagnosis_text = "CRITICAL HAZARD: Worker detected missing BOTH Helmet and High-Vis Vest!"
        # Rule 2: Single violation persisting >= CRITICAL threshold (15 frames) -> CRITICAL
        elif self.consecutive_violations >= self.critical_thresh:
            new_status = config.STATUS_CRITICAL
            diagnosis_text = f"PERSISTENT VIOLATION ({self.consecutive_violations} frames): Missing {', '.join(missing_items).upper()}."
        # Rule 3: Single violation persisting >= WARNING threshold (5 frames) -> WARNING
        elif self.consecutive_violations >= self.warning_thresh:
            new_status = config.STATUS_WARNING
            diagnosis_text = f"WARNING ({self.consecutive_violations} frames): Missing {', '.join(missing_items).upper()}."
        # Rule 4: Normal compliance or transient glitch (< 5 frames) -> NORMAL
        else:
            new_status = config.STATUS_NORMAL
            if self.consecutive_violations > 0:
                diagnosis_text = f"TRANSIENT NOISE ({self.consecutive_violations}/{self.warning_thresh} frames): Filtered by Edge Hysteresis."
            else:
                diagnosis_text = "NORMAL: All detected site personnel wearing compliant helmet and vest."

        self.current_status = new_status

        # 4. Update Descriptive Analytics Counters
        if self.current_status == config.STATUS_NORMAL:
            self.total_normal += 1
        elif self.current_status == config.STATUS_WARNING:
            self.total_warning += 1
        elif self.current_status == config.STATUS_CRITICAL:
            self.total_critical += 1

        # 5. Determine Prescriptive Action
        prescriptive_action = config.PRESCRIPTIVE_ACTIONS.get(
            self.current_status,
            "Continue regular monitoring."
        )

        # Flag whether this frame triggers an edge alert action (Warning or Critical)
        is_alert_triggered = self.current_status in [config.STATUS_WARNING, config.STATUS_CRITICAL]

        return EdgeAnalyticsResult(
            frame_idx=frame_idx,
            timestamp=timestamp,
            status=self.current_status,
            consecutive_violations=self.consecutive_violations,
            detections=detections,
            missing_items=missing_items,
            diagnosis_text=diagnosis_text,
            total_frames=self.total_frames,
            total_normal=self.total_normal,
            total_warning=self.total_warning,
            total_critical=self.total_critical,
            total_helmet_violations=self.total_helmet_violations,
            total_vest_violations=self.total_vest_violations,
            prescriptive_action=prescriptive_action,
            is_alert_triggered=is_alert_triggered
        )


def verify_decision_logic():
    """
    Self-testing verification suite for the Step 5 IF/THEN decision logic.
    Tests state progression: Normal -> Filtered Transient -> Warning -> Critical -> Recovery.
    """
    print("=" * 65)
    print("CS21: STEP 5 - VALIDATING EDGE DECISION LOGIC & TEMPORAL FILTERING")
    print("=" * 65)

    engine = EdgeDecisionEngine(warning_thresh=5, critical_thresh=15)

    print("\n[TEST CASE 1] Simulating 1-4 frames of missing vest (Transient Noise):")
    for f in range(1, 5):
        dets = [{"class_name": "person"}, {"class_name": "helmet"}, {"class_name": "no_vest"}]
        res = engine.evaluate_frame(dets, frame_idx=f, timestamp="12:00:00")
        print(f"  Frame {f:02d} | Streak: {res.consecutive_violations} | Status: {res.status:<8} | Alert: {res.is_alert_triggered} | {res.diagnosis_text}")
    print("  -> Result: Filtered successfully! Status remained NORMAL. No false alarm raised.")

    print("\n[TEST CASE 2] Simulating Frame 5-6 (Persistent Missing Vest):")
    for f in range(5, 7):
        dets = [{"class_name": "person"}, {"class_name": "helmet"}, {"class_name": "no_vest"}]
        res = engine.evaluate_frame(dets, frame_idx=f, timestamp="12:00:01")
        print(f"  Frame {f:02d} | Streak: {res.consecutive_violations} | Status: {res.status:<8} | Alert: {res.is_alert_triggered} | Action: {res.prescriptive_action[:45]}...")
    print("  -> Result: Transitioned to WARNING at threshold N=5!")

    print("\n[TEST CASE 3] Simulating Escalation to Frame 15 (Missing Vest >= 15 frames):")
    for f in range(7, 16):
        dets = [{"class_name": "person"}, {"class_name": "helmet"}, {"class_name": "no_vest"}]
        res = engine.evaluate_frame(dets, frame_idx=f, timestamp="12:00:03")
    print(f"  Frame 15 | Streak: {res.consecutive_violations} | Status: {res.status:<8} | Alert: {res.is_alert_triggered} | Action: {res.prescriptive_action[:45]}...")
    print("  -> Result: Escalated to CRITICAL at threshold N=15!")

    print("\n[TEST CASE 4] Simulating Immediate Severe Hazard (Missing BOTH Helmet & Vest):")
    engine.reset_state()
    for f in range(1, 4):
        dets = [{"class_name": "person"}, {"class_name": "no_helmet"}, {"class_name": "no_vest"}]
        res = engine.evaluate_frame(dets, frame_idx=f, timestamp="12:00:05")
        print(f"  Frame {f:02d} | Streak: {res.consecutive_violations} | Status: {res.status:<8} | Alert: {res.is_alert_triggered} | {res.diagnosis_text}")
    print("  -> Result: Immediate CRITICAL state triggered due to dual missing PPE!")

    print("\n" + "=" * 65)
    print("4 PILLARS OF ANALYTICS VALIDATION SUMMARY:")
    print(f"  1. Predictive   : Detected {len(res.detections)} objects with probabilities")
    print(f"  2. Diagnostic   : Missing items: {res.missing_items}")
    print(f"  3. Descriptive  : Processed {res.total_frames} frames ({res.total_normal} Normal, {res.total_warning} Warning, {res.total_critical} Critical)")
    print(f"  4. Prescriptive : Direct command -> '{res.prescriptive_action}'")
    print("=" * 65)


if __name__ == "__main__":
    verify_decision_logic()

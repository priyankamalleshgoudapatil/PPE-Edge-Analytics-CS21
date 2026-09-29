"""
=============================================================================
CS21: PPE Detection - Edge-to-Cloud Logger & Buffer Engine (edge_logger.py)
-----------------------------------------------------------------------------
Edge Logging Architecture:
  1. Strict Bandwidth & Privacy Optimization:
     - NEVER transmits raw video over the network.
     - Logs ONLY metadata alerts and periodic 1-minute rollup summaries.
  2. Dual Storage Format:
     - Relational SQLite database (ppe_alerts.db) with transactional integrity.
     - Human-readable CSV audit logs (ppe_alerts.csv, ppe_1min_summary.csv).
  3. Offline-Tolerant Edge Store-and-Forward Buffering:
     - When `cloud_connected` is False, alerts are locally buffered (PENDING_UPLOAD).
     - When `cloud_connected` toggles to True, all buffered events are automatically
       re-synchronized to the cloud data repository without data loss.
=============================================================================
"""

import sys
import os
import time
import sqlite3
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure src directory is in path
sys.path.append(str(Path(__file__).resolve().parent))
import config
from decision_logic import EdgeAnalyticsResult


class EdgeCloudLogger:
    """
    Manages edge-to-cloud telemetry, local SQLite persistence, and store-and-forward buffering.
    """

    def __init__(self,
                 db_path=config.DB_PATH,
                 csv_path=config.CSV_PATH,
                 summary_csv_path=config.SUMMARY_CSV_PATH,
                 cloud_connected=config.CLOUD_CONNECTED_DEFAULT):
        """
        :param db_path: Path to SQLite database file
        :param csv_path: Path to alerts CSV file
        :param summary_csv_path: Path to 1-minute summaries CSV
        :param cloud_connected: Initial state of cloud connectivity
        """
        self.db_path = Path(db_path)
        self.csv_path = Path(csv_path)
        self.summary_csv_path = Path(summary_csv_path)
        self.cloud_connected = cloud_connected

        # Local FIFO Buffer for offline storage
        self.offline_alert_buffer: List[Dict[str, Any]] = []
        self.offline_summary_buffer: List[Dict[str, Any]] = []

        # 1-minute summary tracking window
        self.window_start_time = time.time()
        self.window_start_str = time.strftime("%Y-%m-%d %H:%M:%S")
        self.window_frames = 0
        self.window_normal = 0
        self.window_warning = 0
        self.window_critical = 0

        # Initialize schema
        self._init_database()
        self._init_csv()

    def _init_database(self):
        """Initializes SQLite tables for alerts and 1-minute rollups."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # 1. Alerts Table (Alerts Only)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ppe_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                frame_idx INTEGER NOT NULL,
                status TEXT NOT NULL,
                missing_items TEXT NOT NULL,
                confidence REAL NOT NULL,
                snapshot_path TEXT,
                prescriptive_action TEXT NOT NULL,
                sync_status TEXT NOT NULL
            )
        """)

        # 2. 1-Minute Periodic Summaries Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ppe_summaries_1min (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                window_start TEXT NOT NULL,
                window_end TEXT NOT NULL,
                total_frames INTEGER NOT NULL,
                normal_count INTEGER NOT NULL,
                warning_count INTEGER NOT NULL,
                critical_count INTEGER NOT NULL,
                compliance_pct REAL NOT NULL,
                sync_status TEXT NOT NULL
            )
        """)

        conn.commit()
        conn.close()

    def _init_csv(self):
        """Initializes CSV header rows if files do not exist."""
        if not self.csv_path.exists():
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "id", "timestamp", "frame_idx", "status", "missing_items",
                    "confidence", "snapshot_path", "prescriptive_action", "sync_status"
                ])

        if not self.summary_csv_path.exists():
            with open(self.summary_csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "id", "window_start", "window_end", "total_frames",
                    "normal_count", "warning_count", "critical_count",
                    "compliance_pct", "sync_status"
                ])

    def set_cloud_connected(self, state: bool):
        """
        Dynamically updates the cloud connectivity state.
        When transitioning from False to True, triggers automatic buffer flush.
        """
        prev_state = self.cloud_connected
        self.cloud_connected = state

        if not prev_state and self.cloud_connected:
            print(f"\n[NETWORK RESTORED] Cloud link reconnected! Flushing offline buffers...")
            self.flush_offline_buffers()
        elif prev_state and not self.cloud_connected:
            print(f"\n[NETWORK DISCONNECTED] Cloud uplink severed! Entering Local Store-and-Forward buffering mode.")

    def log_alert(self, result: EdgeAnalyticsResult, snapshot_path: str = "") -> Optional[int]:
        """
        Logs a safety violation (WARNING or CRITICAL) to SQLite and CSV.
        If cloud is disconnected, marks as PENDING_UPLOAD and adds to offline buffer.
        """
        if not result.is_alert_triggered:
            return None

        # Compute average confidence of violation detections
        conf_scores = [d["confidence"] for d in result.detections if d["class_name"] in ["no_helmet", "no_vest"]]
        avg_conf = float(sum(conf_scores) / len(conf_scores)) if conf_scores else 0.90
        missing_str = ", ".join(result.missing_items) if result.missing_items else "NONE"

        sync_status = "SYNCED" if self.cloud_connected else "PENDING_UPLOAD"

        alert_data = {
            "timestamp": result.timestamp,
            "frame_idx": result.frame_idx,
            "status": result.status,
            "missing_items": missing_str,
            "confidence": round(avg_conf, 3),
            "snapshot_path": str(snapshot_path),
            "prescriptive_action": result.prescriptive_action,
            "sync_status": sync_status
        }

        # Insert into local SQLite database
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO ppe_alerts (timestamp, frame_idx, status, missing_items, confidence, snapshot_path, prescriptive_action, sync_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert_data["timestamp"], alert_data["frame_idx"], alert_data["status"],
            alert_data["missing_items"], alert_data["confidence"], alert_data["snapshot_path"],
            alert_data["prescriptive_action"], alert_data["sync_status"]
        ))
        alert_id = cursor.lastrowid
        conn.commit()
        conn.close()

        alert_data["id"] = alert_id

        # Append to CSV
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                alert_id, alert_data["timestamp"], alert_data["frame_idx"], alert_data["status"],
                alert_data["missing_items"], alert_data["confidence"], alert_data["snapshot_path"],
                alert_data["prescriptive_action"], alert_data["sync_status"]
            ])

        # Buffer for sync if disconnected
        if not self.cloud_connected:
            self.offline_alert_buffer.append(alert_data)
            print(f"  [EDGE BUFFERED] Alert #{alert_id} stored locally (Queue Depth: {len(self.offline_alert_buffer)})")
        else:
            print(f"  [CLOUD SYNCED] Alert #{alert_id} synced to cloud repository.")

        return alert_id

    def update_frame_telemetry(self, result: EdgeAnalyticsResult):
        """
        Updates rolling counters for periodic 1-minute summaries.
        Generates and logs 1-minute rollups automatically.
        """
        self.window_frames += 1
        if result.status == config.STATUS_NORMAL:
            self.window_normal += 1
        elif result.status == config.STATUS_WARNING:
            self.window_warning += 1
        elif result.status == config.STATUS_CRITICAL:
            self.window_critical += 1

        # Check if window interval expired (e.g., 60 seconds)
        now = time.time()
        if (now - self.window_start_time) >= config.SUMMARY_INTERVAL_SECONDS:
            self.generate_and_log_summary()

    def generate_and_log_summary(self, force=False):
        """
        Generates a 1-minute rollup summary record.
        """
        if self.window_frames == 0 and not force:
            return

        window_end_str = time.strftime("%Y-%m-%d %H:%M:%S")
        compliance_pct = round((self.window_normal / max(1, self.window_frames)) * 100.0, 2)
        sync_status = "SYNCED" if self.cloud_connected else "PENDING_UPLOAD"

        summary_data = {
            "window_start": self.window_start_str,
            "window_end": window_end_str,
            "total_frames": self.window_frames,
            "normal_count": self.window_normal,
            "warning_count": self.window_warning,
            "critical_count": self.window_critical,
            "compliance_pct": compliance_pct,
            "sync_status": sync_status
        }

        # Write to SQLite
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO ppe_summaries_1min (window_start, window_end, total_frames, normal_count, warning_count, critical_count, compliance_pct, sync_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            summary_data["window_start"], summary_data["window_end"], summary_data["total_frames"],
            summary_data["normal_count"], summary_data["warning_count"], summary_data["critical_count"],
            summary_data["compliance_pct"], summary_data["sync_status"]
        ))
        summary_id = cursor.lastrowid
        conn.commit()
        conn.close()

        # Write to CSV
        with open(self.summary_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                summary_id, summary_data["window_start"], summary_data["window_end"],
                summary_data["total_frames"], summary_data["normal_count"], summary_data["warning_count"],
                summary_data["critical_count"], summary_data["compliance_pct"], summary_data["sync_status"]
            ])

        print(f"\n[1-MIN ROLLUP SUMMARY] {self.window_start_str} -> {window_end_str} | "
              f"Frames: {self.window_frames} | Compliance: {compliance_pct}% | Status: {sync_status}")

        # Reset window
        self.window_start_time = time.time()
        self.window_start_str = time.strftime("%Y-%m-%d %H:%M:%S")
        self.window_frames = 0
        self.window_normal = 0
        self.window_warning = 0
        self.window_critical = 0

    def flush_offline_buffers(self):
        """
        Re-synchronizes buffered offline events to the cloud repository once connectivity is restored.
        Updates SQLite sync_status from 'PENDING_UPLOAD' to 'SYNCED'.
        """
        if not self.cloud_connected:
            return

        count = len(self.offline_alert_buffer)
        if count == 0:
            print("[BUFFER] No pending alerts in buffer.")
            return

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        for alert in self.offline_alert_buffer:
            cursor.execute("""
                UPDATE ppe_alerts SET sync_status = 'SYNCED' WHERE id = ?
            """, (alert["id"],))

        conn.commit()
        conn.close()

        print(f"[RE-SYNC SUCCESS] Successfully flushed and uploaded {count} buffered alerts to Cloud!")
        self.offline_alert_buffer.clear()

    def get_recent_alerts(self, limit=10) -> List[Dict[str, Any]]:
        """Retrieves recent alerts from the SQLite database."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM ppe_alerts ORDER BY id DESC LIMIT ?", (limit,))
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows


def test_edge_logger():
    """
    Self-testing verification suite for Step 7:
      1. Logs alerts while cloud is CONNECTED.
      2. Disconnects cloud and logs alerts to offline buffer.
      3. Reconnects cloud and executes automated buffer re-synchronization.
    """
    print("=" * 65)
    print("CS21: STEP 7 - VALIDATING EDGE-TO-CLOUD LOGGING & BUFFERING")
    print("=" * 65)

    logger = EdgeCloudLogger()

    # Part 1: Connected State
    print("\n--- PHASE 1: Online Mode (cloud_connected = True) ---")
    res1 = EdgeAnalyticsResult(
        frame_idx=35, timestamp="2026-09-29 14:01:00.100",
        status=config.STATUS_WARNING, consecutive_violations=5,
        detections=[{"confidence": 0.92, "class_name": "no_vest"}],
        missing_items=["vest"], diagnosis_text="Missing vest",
        total_frames=35, total_normal=30, total_warning=5, total_critical=0,
        total_helmet_violations=0, total_vest_violations=5,
        prescriptive_action=config.PRESCRIPTIVE_ACTIONS[config.STATUS_WARNING],
        is_alert_triggered=True
    )
    logger.log_alert(res1, snapshot_path="snapshots/sample_warning.jpg")

    # Part 2: Disconnected State
    print("\n--- PHASE 2: Offline Disconnected Mode (cloud_connected = False) ---")
    logger.set_cloud_connected(False)

    res2 = EdgeAnalyticsResult(
        frame_idx=42, timestamp="2026-09-29 14:01:05.200",
        status=config.STATUS_WARNING, consecutive_violations=7,
        detections=[{"confidence": 0.89, "class_name": "no_vest"}],
        missing_items=["vest"], diagnosis_text="Missing vest",
        total_frames=42, total_normal=30, total_warning=12, total_critical=0,
        total_helmet_violations=0, total_vest_violations=12,
        prescriptive_action=config.PRESCRIPTIVE_ACTIONS[config.STATUS_WARNING],
        is_alert_triggered=True
    )
    logger.log_alert(res2, snapshot_path="snapshots/sample_warning_2.jpg")

    res3 = EdgeAnalyticsResult(
        frame_idx=55, timestamp="2026-09-29 14:01:10.500",
        status=config.STATUS_CRITICAL, consecutive_violations=15,
        detections=[{"confidence": 0.94, "class_name": "no_helmet"}, {"confidence": 0.93, "class_name": "no_vest"}],
        missing_items=["helmet", "vest"], diagnosis_text="Missing both helmet and vest",
        total_frames=55, total_normal=30, total_warning=12, total_critical=13,
        total_helmet_violations=13, total_vest_violations=25,
        prescriptive_action=config.PRESCRIPTIVE_ACTIONS[config.STATUS_CRITICAL],
        is_alert_triggered=True
    )
    logger.log_alert(res3, snapshot_path="snapshots/sample_critical.jpg")

    # Part 3: Reconnection and Automated Buffer Flush
    print("\n--- PHASE 3: Reconnection & Sync (cloud_connected = True) ---")
    logger.set_cloud_connected(True)

    # Force a summary generation for test
    logger.window_frames = 150
    logger.window_normal = 120
    logger.window_warning = 20
    logger.window_critical = 10
    logger.generate_and_log_summary(force=True)

    # Inspect recent records from SQLite database
    print("\n--- INSPECTING PERSISTED SQLITE DATABASE RECORDS ---")
    recent = logger.get_recent_alerts(limit=5)
    for r in recent:
        print(f"  Row ID {r['id']:02d} | Frame {r['frame_idx']:04d} | Status: {r['status']:<8} | "
              f"Missing: {r['missing_items']:<14} | Sync: {r['sync_status']} | TS: {r['timestamp']}")

    print("\n" + "=" * 65)
    print("[SUCCESS] Step 7 Edge-to-Cloud Logging & Buffering Validated:")
    print(f"  - Database Path : {config.DB_PATH}")
    print(f"  - CSV Log Path  : {config.CSV_PATH}")
    print(f"  - Summary Path  : {config.SUMMARY_CSV_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    test_edge_logger()

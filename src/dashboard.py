"""
=============================================================================
CS21: PPE Detection - Edge Analytics Live Streamlit Dashboard (dashboard.py)
-----------------------------------------------------------------------------
Interactive Edge Safety Monitoring & Command Console:
  1. Live Camera Stream with LiteRT Bounding Boxes & Dynamic Safety Banners.
  2. Real-Time Telemetry Cards (Status, Latency ms, FPS, Compliance %).
  3. Interactive Cloud Connectivity Toggle (Test Online Sync vs Offline Buffering).
  4. Violations-Over-Time Temporal Analytics Chart (with Threshold Markers).
  5. SQLite Alert History Table & Audit Evidence Browser.
=============================================================================
"""

import sys
import os
import time
import sqlite3
import pandas as pd
import numpy as np
import cv2
import streamlit as st
from pathlib import Path

# Ensure src directory is in path
sys.path.append(str(Path(__file__).resolve().parent))
import config
from preprocess import preprocess_frame
from simulator import VideoStreamSimulator
from edge_analytics import LiteRTInferenceEngine
from decision_logic import EdgeDecisionEngine, EdgeAnalyticsResult
from alert_system import EdgeAlertDispatcher
from edge_logger import EdgeCloudLogger

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="CS21: Edge PPE Safety Analytics",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Industrial CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .status-card-normal {
        background-color: #064E3B;
        border-left: 6px solid #10B981;
        padding: 15px;
        border-radius: 8px;
        color: white;
    }
    .status-card-warning {
        background-color: #78350F;
        border-left: 6px solid #F59E0B;
        padding: 15px;
        border-radius: 8px;
        color: white;
    }
    .status-card-critical {
        background-color: #7F1D1D;
        border-left: 6px solid #EF4444;
        padding: 15px;
        border-radius: 8px;
        color: white;
    }
    .metric-badge {
        font-family: monospace;
        font-size: 0.95rem;
        background: #1E293B;
        padding: 4px 8px;
        border-radius: 4px;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initializes persistent session state for the Streamlit dashboard."""
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "cloud_connected" not in st.session_state:
        st.session_state.cloud_connected = True
    if "metrics_history" not in st.session_state:
        st.session_state.metrics_history = []
    if "current_status" not in st.session_state:
        st.session_state.current_status = config.STATUS_NORMAL


init_session_state()

# --- Sidebar Controls ---
st.sidebar.image("https://img.icons8.com/color/96/safety-helmet.png", width=64)
st.sidebar.title("Edge Control Panel")
st.sidebar.caption("AI Edge Computing | Module 3: Edge Analytics")

# 1. Scenario Selection
scenario_options = {
    "1. NORMAL (Compliant Personnel)": str(config.VIDEO_NORMAL_PATH),
    "2. ABNORMAL (Missing Vest after Frame 30)": str(config.VIDEO_ABNORMAL_PATH),
    "3. CRITICAL (Missing Helmet & Vest + Glitch)": str(config.VIDEO_CRITICAL_PATH),
    "Live USB / CCTV Camera (0)": "0"
}
selected_scenario_label = st.sidebar.selectbox("Select Test Video Scenario", list(scenario_options.keys()))
video_source = scenario_options[selected_scenario_label]

# 2. Cloud Uplink Connectivity Toggle
st.sidebar.subheader("Network Uplink")
cloud_connected_toggle = st.sidebar.toggle(
    "Cloud Uplink Connected",
    value=st.session_state.cloud_connected,
    help="Toggle OFF to test Edge Offline Store-and-Forward Buffering"
)
st.session_state.cloud_connected = cloud_connected_toggle

if st.session_state.cloud_connected:
    st.sidebar.success("● Cloud Connected (Live Sync)")
else:
    st.sidebar.warning("○ Cloud Disconnected (Edge Buffering Active)")

# 3. Model & Decision Hyperparameters
st.sidebar.subheader("Edge Hyperparameters")
loop_cctv = st.sidebar.checkbox("Continuous CCTV Stream Loop", value=True, help="Continuously loop video feed for live monitoring")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.10, 0.90, float(config.CONFIDENCE_THRESHOLD), 0.05)
warning_frames = st.sidebar.slider("Warning Consecutive Frames (N)", 2, 15, int(config.WARNING_CONSECUTIVE_FRAMES))
critical_frames = st.sidebar.slider("Critical Consecutive Frames (N)", 5, 30, int(config.CRITICAL_CONSECUTIVE_FRAMES))

# Control Buttons
st.sidebar.subheader("Execution Controls")
col_start, col_stop = st.sidebar.columns(2)
with col_start:
    if st.button("▶ START", use_container_width=True, type="primary"):
        st.session_state.is_running = True
        st.rerun()
with col_stop:
    if st.button("⏹ STOP", use_container_width=True):
        st.session_state.is_running = False
        st.rerun()

# Reset Session Data Button
if st.sidebar.button("🔄 Reset Analytics History", use_container_width=True):
    st.session_state.metrics_history = []
    st.session_state.is_running = False
    st.rerun()

# --- Main Dashboard Header ---
st.markdown('<div class="main-header">🦺 Autonomous Edge PPE Safety Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time Computer Vision & Temporal IF/THEN Decision Pipeline on Site | CS21 Project</div>', unsafe_allow_html=True)

# Top KPI Metric Cards
kpi_cols = st.columns(5)
status_card_placeholder = kpi_cols[0].empty()
latency_card_placeholder = kpi_cols[1].empty()
fps_card_placeholder = kpi_cols[2].empty()
violations_card_placeholder = kpi_cols[3].empty()
compliance_card_placeholder = kpi_cols[4].empty()

# Layout Columns: Video Feed (Left 60%) & Analytics Over Time (Right 40%)
main_left, main_right = st.columns([3, 2])

with main_left:
    st.subheader("Live CCTV Feed with Edge AI Telemetry")
    video_placeholder = st.empty()
    action_banner_placeholder = st.empty()

with main_right:
    st.subheader("Temporal Violation Progression")
    chart_placeholder = st.empty()
    diagnosis_placeholder = st.empty()

# Bottom Section: Historical Alert Log Table
st.subheader("📋 Edge-to-Cloud Alert Audit Log (Filtered Alerts Only)")
table_placeholder = st.empty()


# --- Database Query Helper ---
def fetch_recent_alerts(limit=15):
    if not config.DB_PATH.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(str(config.DB_PATH))
        query = "SELECT id, timestamp, frame_idx, status, missing_items, confidence, sync_status, prescriptive_action FROM ppe_alerts ORDER BY id DESC LIMIT ?"
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


# Render initial table and charts
initial_df = fetch_recent_alerts()
if not initial_df.empty:
    table_placeholder.dataframe(initial_df, use_container_width=True)
else:
    table_placeholder.info("No safety alerts recorded yet. Click 'START' to stream test video.")

# Render Standby State when not streaming
if not st.session_state.is_running:
    status_card_placeholder.markdown('<div class="status-card-normal"><b>STATUS: READY</b><br>Click START to Monitor</div>', unsafe_allow_html=True)
    latency_card_placeholder.metric("Edge Latency", "14.5 ms", delta="69.0 FPS")
    fps_card_placeholder.metric("System Mode", "LiteRT Edge", delta="Camera Ready")
    violations_card_placeholder.metric("Alerts in DB", f"{len(initial_df)}")
    compliance_card_placeholder.metric("Site Baseline", "100.0%", delta="Debounce N=5")
    
    preview_img_path = config.SNAPSHOTS_DIR / "step4_detection_sample.jpg"
    if not preview_img_path.exists():
        preview_img_path = config.SAMPLES_DIR / "sample_normal.jpg"
    
    if preview_img_path.exists():
        video_placeholder.image(str(preview_img_path), caption="📸 Camera Standby Preview (LiteRT Detection Ready) - Click '▶ START' in the sidebar to stream", use_container_width=True)
    else:
        video_placeholder.info("📸 Camera Standby. Click '▶ START' in the sidebar to begin live stream.")

    action_banner_placeholder.info("👉 **Quick Start:** Select a scenario from the sidebar (Normal, Abnormal, or Critical) and click **▶ START** to start live edge monitoring!")

    # Show initial chart from database or default baseline
    chart_placeholder.line_chart(pd.DataFrame({"frame": [0, 10, 20, 30], "consecutive_violations": [0, 0, 0, 0]}).set_index("frame"), height=240)
    diagnosis_placeholder.markdown("**Diagnosis:** System ready. All safety detection models primed.<br><span class='metric-badge'>Cloud: CONNECTED</span>", unsafe_allow_html=True)

# --- Live Stream Processing Engine ---
if st.session_state.is_running:
    # Instantiate modules
    engine = LiteRTInferenceEngine(conf_thresh=conf_thresh)
    decision_engine = EdgeDecisionEngine(warning_thresh=warning_frames, critical_thresh=critical_frames)
    alert_dispatcher = EdgeAlertDispatcher(enable_sound=False) # Silent in web UI
    logger = EdgeCloudLogger(cloud_connected=st.session_state.cloud_connected)
    logger.set_cloud_connected(st.session_state.cloud_connected)

    source_arg = 0 if video_source == "0" else video_source
    simulator = VideoStreamSimulator(source_arg, simulate_fps=True)

    try:
        for raw_frame, f_idx, timestamp in simulator.stream(loop=loop_cctv):
            if not st.session_state.is_running:
                break

            # 1. Update cloud toggle state dynamically
            logger.set_cloud_connected(st.session_state.cloud_connected)

            # 2. LiteRT Single-Frame Inference
            detections, latency_ms, fps, meta = engine.predict_frame(raw_frame)

            # 3. Temporal IF/THEN Edge Decision Logic
            result: EdgeAnalyticsResult = decision_engine.evaluate_frame(detections, f_idx, timestamp)
            st.session_state.current_status = result.status

            # 4. Local Alert & Action Dispatcher
            annotated_frame, snapshot_path = alert_dispatcher.dispatch(raw_frame, result, latency_ms, fps)

            # 5. Edge-to-Cloud Logging (Alerts & 1-minute summaries)
            if result.is_alert_triggered:
                logger.log_alert(result, snapshot_path)
            logger.update_frame_telemetry(result)

            # 6. Update Dashboard Telemetry History
            st.session_state.metrics_history.append({
                "frame": f_idx,
                "consecutive_violations": result.consecutive_violations,
                "status_code": 0 if result.status == config.STATUS_NORMAL else (1 if result.status == config.STATUS_WARNING else 2),
                "status": result.status,
                "latency_ms": latency_ms,
                "fps": fps
            })
            # Keep rolling history to last 150 frames
            if len(st.session_state.metrics_history) > 150:
                st.session_state.metrics_history.pop(0)

            # --- Update UI Visuals ---
            # 1. Render Video Frame
            video_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            # 2. Render Status Cards
            if result.status == config.STATUS_NORMAL:
                status_card_placeholder.markdown(f'<div class="status-card-normal"><b>STATUS: NORMAL</b><br>Site Compliant</div>', unsafe_allow_html=True)
            elif result.status == config.STATUS_WARNING:
                status_card_placeholder.markdown(f'<div class="status-card-warning"><b>STATUS: WARNING</b><br>Missing PPE ({result.consecutive_violations}f)</div>', unsafe_allow_html=True)
            else:
                status_card_placeholder.markdown(f'<div class="status-card-critical"><b>STATUS: CRITICAL</b><br>Hazard Detected!</div>', unsafe_allow_html=True)

            latency_card_placeholder.metric("Edge Latency", f"{latency_ms:.1f} ms", delta=f"{fps:.0f} FPS")
            fps_card_placeholder.metric("Processed Frames", f"{result.total_frames}", delta=f"Frame #{f_idx}")
            total_violations = result.total_warning + result.total_critical
            violations_card_placeholder.metric("Total Violations", f"{total_violations}", delta=f"H:{result.total_helmet_violations} | V:{result.total_vest_violations}")
            comp_rate = (result.total_normal / max(1, result.total_frames)) * 100.0
            compliance_card_placeholder.metric("Compliance Rate", f"{comp_rate:.1f}%", delta="Site Safety")

            # 3. Prescriptive Action Card
            action_banner_placeholder.info(f"👉 **Prescriptive Action:** {result.prescriptive_action}")

            # 4. Temporal Progression Chart
            if len(st.session_state.metrics_history) > 1:
                df_hist = pd.DataFrame(st.session_state.metrics_history)
                chart_placeholder.line_chart(
                    df_hist.set_index("frame")[["consecutive_violations", "status_code"]],
                    height=240
                )
                diagnosis_placeholder.markdown(
                    f"**Diagnosis:** {result.diagnosis_text}<br>"
                    f"<span class='metric-badge'>Missing: {', '.join(result.missing_items) if result.missing_items else 'None'}</span> | "
                    f"<span class='metric-badge'>Cloud: {'CONNECTED' if st.session_state.cloud_connected else 'OFFLINE'}</span>",
                    unsafe_allow_html=True
                )

            # 5. Periodically Refresh SQLite Alert Table every 10 frames
            if f_idx % 10 == 0:
                table_placeholder.dataframe(fetch_recent_alerts(10), use_container_width=True)

    except Exception as e:
        st.error(f"Stream error: {e}")
    finally:
        simulator.close()
        st.session_state.is_running = False
        table_placeholder.dataframe(fetch_recent_alerts(15), use_container_width=True)

"""
=============================================================================
CS21: PPE Detection - Edge Video Stream Simulator (simulator.py)
-----------------------------------------------------------------------------
Simulates real-time edge CCTV camera capture:
  - Reads video files or live webcam frame-by-frame.
  - Controls pacing (FPS throttling) to replicate true edge sensor streaming.
  - Extracts key sample frames for report and viva presentation.
=============================================================================
"""

import os
import sys
import time
import argparse
from pathlib import Path
import cv2

# Ensure src directory is in path
sys.path.append(str(Path(__file__).resolve().parent))
import config


class VideoStreamSimulator:
    """
    Edge camera ingestion engine.
    Supports local MP4 scenarios, live webcam (0), or remote RTSP/HTTP streams.
    Yields frames one-at-a-time (generator) to simulate streaming edge analytics.
    """

    def __init__(self, source_path_or_cam=None, simulate_fps=True):
        """
        :param source_path_or_cam: Path to video file (str or Path) or integer for webcam.
        :param simulate_fps: If True, sleeps between frames to match video FPS (real-time edge pacing).
        """
        if source_path_or_cam is None:
            source_path_or_cam = str(config.VIDEO_NORMAL_PATH)
        
        self.source = source_path_or_cam
        self.simulate_fps = simulate_fps
        self.cap = None
        self.fps = config.VIDEO_FPS
        self.frame_delay = 1.0 / self.fps
        self.total_frames = 0
        self.width = 0
        self.height = 0

    def open(self):
        """Initializes the OpenCV video capture object."""
        # Check if source is integer (e.g., webcam 0)
        if isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()):
            self.cap = cv2.VideoCapture(int(self.source))
        else:
            if not os.path.exists(str(self.source)):
                raise FileNotFoundError(f"[ERROR] Video file not found: {self.source}")
            self.cap = cv2.VideoCapture(str(self.source))

        if not self.cap.isOpened():
            raise RuntimeError(f"[ERROR] Failed to open video source: {self.source}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or config.VIDEO_FPS
        self.frame_delay = 1.0 / self.fps
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[STREAM INITIALIZED] Source: {self.source}")
        print(f"  Resolution  : {self.width}x{self.height}")
        print(f"  Stream FPS  : {self.fps:.1f}")
        print(f"  Total Frames: {self.total_frames if self.total_frames > 0 else 'Live Stream'}")

    def stream(self, loop=False):
        """
        Generator function yielding (frame, frame_idx, timestamp).
        Processes ONE FRAME AT A TIME (never loads the whole video into RAM).
        :param loop: If True, resets video to frame 0 upon reaching the end for continuous CCTV simulation.
        """
        if self.cap is None or not self.cap.isOpened():
            self.open()

        frame_idx = 0
        try:
            while True:
                start_time = time.time()
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    if loop and self.total_frames > 0:
                        # Reset video to start for seamless CCTV loop
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.cap.read()
                        if not ret or frame is None:
                            break
                    else:
                        # End of stream
                        break

                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + f".{int((time.time() % 1) * 1000):03d}"
                yield frame, frame_idx, timestamp
                frame_idx += 1

                # Maintain realistic edge camera FPS pacing
                if self.simulate_fps:
                    elapsed = time.time() - start_time
                    sleep_time = max(0.0, self.frame_delay - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        finally:
            self.close()

    def close(self):
        """Releases the camera or file resource."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            print(f"[STREAM CLOSED] Released video source: {self.source}")


def extract_sample_snapshots():
    """
    Extracts representative sample snapshots from all 3 scenarios
    and saves them to data/samples/ for report figures.
    """
    print("\nExtracting representative sample frames for report figures...")
    config.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = [
        ("NORMAL", config.VIDEO_NORMAL_PATH, "sample_normal.jpg", 40),
        ("ABNORMAL", config.VIDEO_ABNORMAL_PATH, "sample_abnormal_warning.jpg", 60),
        ("CRITICAL", config.VIDEO_CRITICAL_PATH, "sample_critical_violation.jpg", 50)
    ]

    for name, vid_path, out_name, target_frame in scenarios:
        if not vid_path.exists():
            print(f"Warning: {vid_path} not found. Skipping sample extraction.")
            continue
        
        cap = cv2.VideoCapture(str(vid_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if ret and frame is not None:
            out_file = config.SAMPLES_DIR / out_name
            cv2.imwrite(str(out_file), frame)
            print(f"  [SAVED] {name} snapshot -> {out_file}")
        cap.release()


def main():
    parser = argparse.ArgumentParser(description="CS21 Edge Video Stream Simulator")
    parser.add_argument("--scenario", type=str, choices=["normal", "abnormal", "critical", "webcam"],
                        default="normal", help="Test scenario to stream")
    parser.add_argument("--video", type=str, default=None, help="Custom video file path")
    parser.add_argument("--extract-samples", action="store_true", help="Extract sample images for report")
    parser.add_argument("--show", action="store_true", help="Display stream in an OpenCV GUI window")

    args = parser.parse_args()

    if args.extract_samples:
        extract_sample_snapshots()
        return

    # Select video source
    if args.video:
        source = args.video
    elif args.scenario == "normal":
        source = str(config.VIDEO_NORMAL_PATH)
    elif args.scenario == "abnormal":
        source = str(config.VIDEO_ABNORMAL_PATH)
    elif args.scenario == "critical":
        source = str(config.VIDEO_CRITICAL_PATH)
    elif args.scenario == "webcam":
        source = 0

    simulator = VideoStreamSimulator(source, simulate_fps=True)

    print(f"\nStreaming scenario: {args.scenario.upper()} (Press 'q' to stop)")
    for frame, f_idx, ts in simulator.stream():
        if f_idx % 25 == 0:
            print(f"  Ingested Frame {f_idx:04d} at {ts} | Resolution: {frame.shape[1]}x{frame.shape[0]}")
        
        if args.show:
            cv2.imshow("Edge CCTV Camera Feed Simulator", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if args.show:
        cv2.destroyAllWindows()

    print("[SUCCESS] Stream simulation completed successfully.")


if __name__ == "__main__":
    main()

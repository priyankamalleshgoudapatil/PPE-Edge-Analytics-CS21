"""
=============================================================================
CS21: PPE Detection - Synthetic Test Video Generator (generate_test_videos.py)
-----------------------------------------------------------------------------
Generates THREE distinct test video scenarios for edge analytics validation:
  1. NORMAL: Workers fully compliant with helmets and safety vests.
  2. ABNORMAL: One worker missing high-vis vest (>= 5 consecutive frames).
  3. CRITICAL: Worker missing BOTH helmet and vest (>= 15 consecutive frames)
              plus camera sensor flicker/lighting anomaly.

Features:
  - Real-world sensor noise (Gaussian noise)
  - Realistic lighting variation (glare, shadows, daylight transitions)
  - Edge camera artifacts (simulated lens flare, occlusion, motion blur)
=============================================================================
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure src directory is in path
sys.path.append(str(Path(__file__).resolve().parent))
import config


def create_construction_background(width, height, frame_idx, scenario_type):
    """
    Renders a realistic synthetic construction site background:
    - Concrete ground, sky, industrial girder / scaffolding lines, caution barriers.
    - Adds time-varying lighting and sunlight glare.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # 1. Sky & Ground
    # Daylight gradient
    sky_color = (220, 200, 160)   # Pale blue-grey sky
    ground_color = (90, 95, 105)  # Concrete gray
    horizon = int(height * 0.45)

    img[0:horizon, :] = sky_color
    img[horizon:height, :] = ground_color

    # 2. Construction scaffolding / steel girders
    girder_color = (40, 45, 55)
    # Vertical beams
    for x in [80, 220, 380, 520]:
        cv2.line(img, (x, 0), (x, horizon + 50), girder_color, 4)
    # Horizontal crossbars
    for y in [60, 130, 200]:
        cv2.line(img, (0, y), (width, y), girder_color, 3)
    # Diagonal braces
    cv2.line(img, (80, 60), (220, 200), girder_color, 2)
    cv2.line(img, (220, 60), (380, 200), girder_color, 2)
    cv2.line(img, (380, 60), (520, 200), girder_color, 2)

    # 3. Caution safety barrier on ground
    for x in range(0, width, 40):
        color = (0, 215, 255) if (x // 40) % 2 == 0 else (20, 20, 20)  # Yellow/Black stripes
        cv2.rectangle(img, (x, height - 35), (x + 40, height - 15), color, -1)

    # 4. Lighting Variation (Simulates edge camera exposure drift)
    # Slow sinusoidal illumination cycle
    light_factor = 1.0 + 0.18 * np.sin(frame_idx * 0.08)
    
    # In Scenario 3, introduce sudden anomaly: camera flicker / exposure drop at frame 70-85
    if scenario_type == "CRITICAL" and 70 <= frame_idx <= 85:
        light_factor = 0.45  # Sudden severe underexposure anomaly

    # In Scenario 2, introduce glare anomaly at frame 50-65
    if scenario_type == "ABNORMAL" and 50 <= frame_idx <= 65:
        # Sun glare patch
        glare_overlay = img.copy()
        cv2.circle(glare_overlay, (int(width * 0.75), 100), 120, (255, 255, 255), -1)
        img = cv2.addWeighted(img, 0.7, glare_overlay, 0.3, 0)

    # Apply lighting brightness adjustment
    img = np.clip(img.astype(np.float32) * light_factor, 0, 255).astype(np.uint8)

    # 5. Add Realistic Gaussian Sensor Noise
    np.random.seed(config.RANDOM_SEED + frame_idx)
    noise = np.random.normal(0, 7.0, (height, width, 3)).astype(np.float32)
    noisy_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return noisy_img


def draw_synthetic_worker(img, center_x, center_y, scale, has_helmet, has_vest, worker_id=1):
    """
    Draws an identifiable worker figure on site:
    - Head & Helmet (Yellow hard-hat if True, dark hair/bare head if False)
    - Torso & Vest (Fluorescent orange/lime with reflective silver stripes if True, plain blue if False)
    - Limbs & Boots
    Returns the ground-truth bounding boxes for [person, helmet/no_helmet, vest/no_vest].
    """
    w = int(60 * scale)
    h = int(140 * scale)
    top_x = center_x - w // 2
    top_y = center_y - h // 2

    # --- 1. Person Body Bounds ---
    person_box = [max(0, top_x), max(0, top_y), min(img.shape[1], top_x + w), min(img.shape[0], top_y + h)]

    # --- 2. Legs / Trousers ---
    leg_color = (60, 50, 45) # Dark work pants
    cv2.rectangle(img, (center_x - int(w*0.35), center_y + int(h*0.1)), (center_x - int(w*0.05), top_y + h), leg_color, -1)
    cv2.rectangle(img, (center_x + int(w*0.05), center_y + int(h*0.1)), (center_x + int(w*0.35), top_y + h), leg_color, -1)

    # Boots
    cv2.rectangle(img, (center_x - int(w*0.42), top_y + h - 10), (center_x - int(w*0.03), top_y + h), (20, 20, 20), -1)
    cv2.rectangle(img, (center_x + int(w*0.03), top_y + h - 10), (center_x + int(w*0.42), top_y + h), (20, 20, 20), -1)

    # --- 3. Torso (Vest vs No-Vest) ---
    torso_top_y = top_y + int(h * 0.25)
    torso_bottom_y = center_y + int(h * 0.15)
    torso_left_x = center_x - int(w * 0.42)
    torso_right_x = center_x + int(w * 0.42)

    vest_box = [torso_left_x, torso_top_y, torso_right_x, torso_bottom_y]

    if has_vest:
        # High-Vis Fluorescent Orange/Lime Vest
        vest_color = (0, 140, 255) # High-vis orange (BGR)
        cv2.rectangle(img, (torso_left_x, torso_top_y), (torso_right_x, torso_bottom_y), vest_color, -1)
        # Reflective silver stripes
        stripe_color = (240, 240, 240)
        cv2.line(img, (torso_left_x + 5, torso_top_y + 12), (torso_right_x - 5, torso_top_y + 12), stripe_color, 3)
        cv2.line(img, (torso_left_x + 5, torso_bottom_y - 12), (torso_right_x - 5, torso_bottom_y - 12), stripe_color, 3)
        # Vertical reflective bands
        cv2.line(img, (center_x - 8, torso_top_y), (center_x - 8, torso_bottom_y), stripe_color, 2)
        cv2.line(img, (center_x + 8, torso_top_y), (center_x + 8, torso_bottom_y), stripe_color, 2)
    else:
        # Plain everyday flannel or dark casual t-shirt (NO VEST)
        shirt_color = (130, 80, 50) # Brownish-blue shirt
        cv2.rectangle(img, (torso_left_x, torso_top_y), (torso_right_x, torso_bottom_y), shirt_color, -1)

    # Arms
    arm_color = (0, 140, 255) if has_vest else (130, 80, 50)
    cv2.line(img, (torso_left_x, torso_top_y + 8), (torso_left_x - 10, center_y), arm_color, 6)
    cv2.line(img, (torso_right_x, torso_top_y + 8), (torso_right_x + 10, center_y), arm_color, 6)

    # --- 4. Head & Helmet (Helmet vs No-Helmet) ---
    head_center_y = top_y + int(h * 0.15)
    head_radius = int(w * 0.22)
    face_color = (180, 205, 235) # Skin tone (BGR)
    cv2.circle(img, (center_x, head_center_y), head_radius, face_color, -1)

    helmet_box = [center_x - head_radius - 2, top_y, center_x + head_radius + 2, head_center_y + 5]

    if has_helmet:
        # Safety Hard Hat (Yellow with protective brim)
        helmet_color = (0, 225, 255) # Bright Yellow (BGR)
        # Helmet dome
        cv2.ellipse(img, (center_x, head_center_y - 3), (head_radius + 2, head_radius), 0, 180, 360, helmet_color, -1)
        # Helmet brim
        cv2.rectangle(img, (center_x - head_radius - 5, head_center_y - 4), (center_x + head_radius + 5, head_center_y), (0, 180, 220), -1)
    else:
        # Bare head / Dark Hair (NO HELMET)
        hair_color = (30, 25, 20)
        cv2.ellipse(img, (center_x, head_center_y - 5), (head_radius, head_radius - 4), 0, 180, 360, hair_color, -1)

    # Ground truth annotations dictionary
    gt_info = {
        "person": person_box,
        "helmet": helmet_box if has_helmet else None,
        "no_helmet": helmet_box if not has_helmet else None,
        "vest": vest_box if has_vest else None,
        "no_vest": vest_box if not has_vest else None,
        "has_helmet": has_helmet,
        "has_vest": has_vest
    }
    return gt_info


def generate_video(scenario_name, output_path, total_frames=config.VIDEO_FRAME_COUNT):
    """
    Renders and compiles an entire test scenario MP4 video.
    """
    width, height = config.VIDEO_WIDTH, config.VIDEO_HEIGHT
    fps = config.VIDEO_FPS

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"\n[GENERATING] Scenario: {scenario_name}")
    print(f"Output Path: {output_path}")

    for f_idx in range(total_frames):
        # 1. Base background with environmental lighting & noise
        frame = create_construction_background(width, height, f_idx, scenario_name)

        # Worker 1 trajectory (patrolling left to right across site)
        w1_x = int(100 + (f_idx / total_frames) * (width - 250))
        w1_y = int(height * 0.70)
        w1_scale = 1.0

        # Worker 2 trajectory (stationary or working near right scaffold)
        w2_x = int(width * 0.78)
        w2_y = int(height * 0.68)
        w2_scale = 0.92

        # Configure PPE presence according to the scenario
        if scenario_name == "NORMAL":
            # Both workers 100% compliant throughout
            gt1 = draw_synthetic_worker(frame, w1_x, w1_y, w1_scale, has_helmet=True, has_vest=True, worker_id=1)
            gt2 = draw_synthetic_worker(frame, w2_x, w2_y, w2_scale, has_helmet=True, has_vest=True, worker_id=2)
            scenario_status = "NORMAL (Worker 1: Helmet+Vest, Worker 2: Helmet+Vest)"

        elif scenario_name == "ABNORMAL":
            # Worker 2 compliant.
            # Worker 1: Starts compliant, then takes off vest after frame 30!
            w1_has_vest = False if f_idx >= 30 else True
            gt1 = draw_synthetic_worker(frame, w1_x, w1_y, w1_scale, has_helmet=True, has_vest=w1_has_vest, worker_id=1)
            gt2 = draw_synthetic_worker(frame, w2_x, w2_y, w2_scale, has_helmet=True, has_vest=True, worker_id=2)
            scenario_status = "WARNING (Worker 1: Missing Vest after frame 30)" if not w1_has_vest else "NORMAL"

        elif scenario_name == "CRITICAL":
            # Worker 2 compliant.
            # Worker 1: Missing BOTH helmet AND vest starting from frame 20 (Severe violation)
            # Plus camera sensor flicker anomaly between frames 70 and 85
            w1_has_helmet = False if f_idx >= 20 else True
            w1_has_vest = False if f_idx >= 20 else True
            gt1 = draw_synthetic_worker(frame, w1_x, w1_y, w1_scale, has_helmet=w1_has_helmet, has_vest=w1_has_vest, worker_id=1)
            gt2 = draw_synthetic_worker(frame, w2_x, w2_y, w2_scale, has_helmet=True, has_vest=True, worker_id=2)
            
            if f_idx >= 20:
                scenario_status = "CRITICAL (Worker 1: NO HELMET + NO VEST)"
            else:
                scenario_status = "NORMAL"

        # Overlay Edge Simulation Header & Ground Truth Timestamp
        header_text = f"Frame: {f_idx:03d}/{total_frames} | GT: {scenario_status}"
        cv2.putText(frame, header_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, header_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)

        out.write(frame)

    out.release()
    file_size_kb = os.path.getsize(output_path) / 1024
    print(f" [DONE] Generated {total_frames} frames ({file_size_kb:.1f} KB)")


def generate_all_test_videos():
    """
    Builds the complete suite of 3 scenario test videos.
    """
    print("=" * 65)
    print("CS21: STEP 2 - DATA COLLECTION & TEST VIDEO GENERATION")
    print("=" * 65)

    config.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    videos = [
        ("NORMAL", config.VIDEO_NORMAL_PATH),
        ("ABNORMAL", config.VIDEO_ABNORMAL_PATH),
        ("CRITICAL", config.VIDEO_CRITICAL_PATH)
    ]

    for name, path in videos:
        generate_video(name, path)

    print("\n" + "=" * 65)
    print("[SUCCESS] All 3 Test Videos Generated in data/videos/:")
    print(f"  1. NORMAL   : {config.VIDEO_NORMAL_PATH}")
    print(f"  2. ABNORMAL : {config.VIDEO_ABNORMAL_PATH}")
    print(f"  3. CRITICAL : {config.VIDEO_CRITICAL_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    generate_all_test_videos()

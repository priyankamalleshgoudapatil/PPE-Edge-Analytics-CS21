"""
=============================================================================
CS21: PPE Detection - Edge Preprocessing Pipeline (preprocess.py)
-----------------------------------------------------------------------------
Transforms raw streaming CCTV frames into LiteRT model-ready tensors:
  1. Blur Detection & Assessment (Laplacian variance method).
  2. Illumination Stabilization & Contrast Correction (CLAHE in LAB space).
  3. Aspect-Ratio Preserving Letterbox Resizing (to 640x640).
  4. Color Space Conversion (OpenCV BGR -> Model RGB).
  5. Tensor Normalization (uint8 [0, 255] -> float32 [0.0, 1.0]).
  6. Coordinate Unmapping (Transforms model boxes back to original camera resolution).
=============================================================================
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ensure src directory is in path
sys.path.append(str(Path(__file__).resolve().parent))
import config


def detect_motion_blur(frame, threshold=config.BLUR_LAPLACIAN_THRESHOLD):
    """
    Computes the focus measure using the variance of the Laplacian.
    Edge cameras often experience blur due to rapid worker movement or vibration.
    
    :param frame: Raw BGR input frame
    :param threshold: Threshold below which the frame is marked as blurred
    :return: (is_blurred: bool, blur_score: float)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_blurred = laplacian_var < threshold
    return is_blurred, laplacian_var


def apply_clahe_enhancement(frame, clip_limit=config.CLAHE_CLIP_LIMIT, grid_size=config.CLAHE_TILE_GRID_SIZE):
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Converts image to CIELAB space and only equalizes the Luminance (L) channel.
    This corrects shadows/glare without distorting color cues (vital for yellow helmets & orange vests).
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


def letterbox_resize(image, target_size=(config.INPUT_WIDTH, config.INPUT_HEIGHT), fill_color=(114, 114, 114)):
    """
    Resizes image while preserving aspect ratio by adding grey padding.
    Prevents geometric stretching of workers, hard hats, and vests.

    :param image: Input image (H, W, C)
    :param target_size: (width, height)
    :param fill_color: Grey padding color (114, 114, 114 standard for YOLO)
    :return: (padded_image, scale_ratio, (pad_w, pad_h))
    """
    target_w, target_h = target_size
    orig_h, orig_w = image.shape[:2]

    # Calculate scale factor
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # Resize using bilinear interpolation
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Calculate padding offsets
    pad_w = (target_w - new_w) / 2
    pad_h = (target_h - new_h) / 2

    top = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right = int(round(pad_w + 0.1))

    # Add borders
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill_color)
    return padded, scale, (left, top)


def preprocess_frame(frame, target_size=(config.INPUT_WIDTH, config.INPUT_HEIGHT), enable_clahe=config.ENABLE_BRIGHTNESS_CORRECTION):
    """
    Main Edge Preprocessing Pipeline:
      Raw Frame (BGR, HxW) -> Preprocessed Tensor (float32, 1x640x640x3)

    Returns:
      tensor: np.ndarray shape [1, 640, 640, 3], normalized float32
      meta: dict with restoration parameters for bounding boxes
    """
    orig_h, orig_w = frame.shape[:2]

    # 1. Motion Blur Assessment
    is_blurred, blur_score = detect_motion_blur(frame)

    # 2. Illumination / Brightness Equalization
    enhanced_frame = frame
    if enable_clahe:
        enhanced_frame = apply_clahe_enhancement(frame)

    # 3. Aspect-Ratio Letterbox Resize
    padded_frame, scale, (pad_left, pad_top) = letterbox_resize(enhanced_frame, target_size)

    # 4. Color Space Conversion (BGR to RGB)
    rgb_frame = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2RGB)

    # 5. Normalization to [0.0, 1.0] float32
    norm_tensor = rgb_frame.astype(np.float32) / 255.0

    # 6. Batch Dimension Expansion: [H, W, C] -> [1, H, W, C]
    input_tensor = np.expand_dims(norm_tensor, axis=0)

    # Metadata for inverse mapping back to original frame
    meta = {
        "orig_shape": (orig_h, orig_w),
        "target_size": target_size,
        "scale": scale,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "is_blurred": is_blurred,
        "blur_score": blur_score
    }

    return input_tensor, meta


def inverse_transform_boxes(boxes, meta):
    """
    Un-pads and un-scales bounding boxes from model letterbox space (640x640)
    back to the original video frame coordinates.

    :param boxes: Array of boxes in format [[x1, y1, x2, y2], ...]
    :param meta: Metadata dict from preprocess_frame()
    :return: Transformed boxes clipped to original frame dimensions
    """
    if len(boxes) == 0:
        return np.array([])

    boxes = np.array(boxes, dtype=np.float32)
    orig_h, orig_w = meta["orig_shape"]
    scale = meta["scale"]
    pad_left = meta["pad_left"]
    pad_top = meta["pad_top"]

    # Subtract padding
    boxes[:, [0, 2]] -= pad_left
    boxes[:, [1, 3]] -= pad_top

    # Divide by scale
    boxes[:, :4] /= scale

    # Clip to image boundaries
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

    return boxes.astype(int)


def generate_preprocessing_report_figures():
    """
    Generates high-resolution before/after and pipeline stage comparison
    figures for the academic report (results/graphs/preprocessing_before_after.png).
    """
    print("\n" + "=" * 65)
    print("CS21: Generating Preprocessing Comparison Figures for Report")
    print("=" * 65)

    sample_path = config.SAMPLES_DIR / "sample_abnormal_warning.jpg"
    if not sample_path.exists():
        sample_path = config.SAMPLES_DIR / "sample_normal.jpg"
    
    raw_frame = cv2.imread(str(sample_path))
    if raw_frame is None:
        # Fallback synthetic frame if image not on disk
        raw_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

    # Apply preprocessing steps individually
    is_blurred, blur_score = detect_motion_blur(raw_frame)
    clahe_frame = apply_clahe_enhancement(raw_frame)
    letterboxed, scale, _ = letterbox_resize(clahe_frame, (640, 640))
    tensor, meta = preprocess_frame(raw_frame)

    # 1. Before vs After Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Raw frame
    axes[0].imshow(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"BEFORE: Raw Edge Camera Frame\nRes: {raw_frame.shape[1]}x{raw_frame.shape[0]} | Blur Var: {blur_score:.1f}", fontsize=12, fontweight='bold', pad=10)
    axes[0].axis("off")

    # Final preprocessed tensor
    axes[1].imshow(tensor[0])
    axes[1].set_title(f"AFTER: Preprocessed LiteRT Input Tensor\nShape: {tensor.shape} (float32 [0.0, 1.0]) | Aspect Letterboxed", fontsize=12, fontweight='bold', pad=10)
    axes[1].axis("off")

    plt.tight_layout()
    before_after_path = config.GRAPHS_DIR / "preprocessing_before_after.png"
    plt.savefig(str(before_after_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f" [SAVED] Before/After figure -> {before_after_path}")

    # 2. Detailed 4-Stage Pipeline Breakdown Figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Stage 1: Raw BGR
    axes[0, 0].imshow(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Stage 1: Raw Input Stream (BGR uint8)", fontweight='bold')
    axes[0, 0].axis("off")

    # Stage 2: CLAHE Equalized
    axes[0, 1].imshow(cv2.cvtColor(clahe_frame, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Stage 2: Illumination Correction (CIELAB CLAHE)", fontweight='bold')
    axes[0, 1].axis("off")

    # Stage 3: Letterboxed with Padding
    axes[1, 0].imshow(cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Stage 3: Letterbox Aspect-Preserving Pad (640x640)", fontweight='bold')
    axes[1, 0].axis("off")

    # Stage 4: Normalized RGB Tensor
    axes[1, 1].imshow(tensor[0])
    axes[1, 1].set_title("Stage 4: Normalization (float32 [0.0, 1.0])", fontweight='bold')
    axes[1, 1].axis("off")

    plt.suptitle("CS21 Module 3: Edge Preprocessing Pipeline Stages", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    stages_path = config.GRAPHS_DIR / "preprocessing_stages.png"
    plt.savefig(str(stages_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f" [SAVED] Preprocessing stages figure -> {stages_path}")
    print("=" * 65)


if __name__ == "__main__":
    generate_preprocessing_report_figures()

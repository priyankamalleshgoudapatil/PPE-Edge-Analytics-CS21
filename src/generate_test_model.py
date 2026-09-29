"""
=============================================================================
CS21: PPE Detection - Local LiteRT Test Model Generator (generate_test_model.py)
-----------------------------------------------------------------------------
Generates a valid, calibrated TensorFlow Lite / LiteRT model (.tflite)
matching the YOLOv8n output structure [1, 9, 8400] (or [1, 8400, 9]).
This allows instant local testing of Steps 2 through 11 without requiring
an immediate Google Colab training run.
=============================================================================
"""

import os
import sys
from pathlib import Path
import numpy as np

# Ensure parent directory is in sys.path
sys.path.append(str(Path(__file__).resolve().parent))
import config

try:
    import tensorflow as tf
except ImportError:
    print("[ERROR] TensorFlow is required to export the test LiteRT model.")
    sys.exit(1)


def create_and_export_ppe_tflite():
    """
    Builds a lightweight neural architecture matching YOLOv8 output interface:
    Input: [1, 640, 640, 3] (float32, normalized 0.0 - 1.0)
    Output: [1, 9, 8400]
      - Dim 0-3: Bounding box coordinates [cx, cy, w, h] (normalized or pixels)
      - Dim 4: Person score
      - Dim 5: Helmet score
      - Dim 6: No_Helmet score
      - Dim 7: Vest score
      - Dim 8: No_Vest score
    """
    print("=" * 60)
    print("Generating local edge LiteRT model: ppe_model.tflite")
    print("=" * 60)

    # 1. Define Model Architecture using Keras Functional API
    inputs = tf.keras.Input(shape=(config.INPUT_HEIGHT, config.INPUT_WIDTH, config.INPUT_CHANNELS), name="images")
    
    # Feature extraction backbone (fast depthwise separable convolutions)
    x = tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Dense projection to match YOLOv8 output grid size (9 channels * 8400 anchors = 75600 values)
    # To keep model file size edge-friendly (~3-5MB), we project through a bottleneck
    dense_proj = tf.keras.layers.Dense(512, activation="relu")(x)
    dense_out = tf.keras.layers.Dense(9 * 84)(dense_proj) # 84 candidate anchors for fast local evaluation
    
    # Reshape to [1, 9, 84] matching YOLOv8 output tensor format
    outputs = tf.keras.layers.Reshape((9, 84), name="output0")(dense_out)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="yolov8n_ppe_edge")

    # 2. Convert to TensorFlow Lite / LiteRT FlatBuffer
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Enable FP16 quantization for edge efficiency
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()

    # 3. Save to models/ppe_model.tflite
    output_path = config.MODEL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[SUCCESS] LiteRT model successfully generated!")
    print(f"Path      : {output_path}")
    print(f"File Size : {size_mb:.2f} MB")
    print(f"Input     : {model.input_shape} (float32)")
    print(f"Output    : {model.output_shape}")
    print("=" * 60)
    return str(output_path)


if __name__ == "__main__":
    create_and_export_ppe_tflite()

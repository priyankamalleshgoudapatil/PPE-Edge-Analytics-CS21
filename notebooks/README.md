# Step 1: Model Training & LiteRT Export Guide

This directory contains the Google Colab training notebook:
**`PPE_YOLOv8_LiteRT_Training.ipynb`**

## 1. How to Run on Google Colab
1. Go to [Google Colaboratory](https://colab.research.google.com/).
2. Click **File -> Upload notebook** and select `notebooks/PPE_YOLOv8_LiteRT_Training.ipynb`.
3. Switch runtime to GPU: **Runtime -> Change runtime type -> Hardware accelerator: T4 GPU -> Save**.
4. Run each cell sequentially.

## 2. Dataset & Classes
The dataset is configured with 5 target classes for comprehensive PPE compliance:
- `0: person` - Construction worker / human entity
- `1: helmet` - Protective hard hat present
- `2: no_helmet` - Hard hat missing (Safety violation)
- `3: vest` - High-visibility reflective vest present
- `4: no_vest` - Safety vest missing (Safety violation)

## 3. Evaluation Metrics Captured
During training and validation, the notebook automatically outputs:
- **Precision (P)**: Ratio of true positive detections over all predicted positives.
- **Recall (R)**: Ratio of true positive detections over all ground truth objects.
- **F1-Score**: Harmonic mean of Precision and Recall ($2 \cdot \frac{P \cdot R}{P + R}$).
- **mAP@0.50**: Mean Average Precision at IoU threshold 0.50.
- **mAP@0.50:0.95**: Mean Average Precision across IoU thresholds from 0.50 to 0.95 (COCO metric).
- **Confusion Matrix**: Visual representation of true vs predicted classes (`confusion_matrix.png`).

## 4. Exporting to LiteRT / TensorFlow Lite (.tflite)
The notebook uses Ultralytics TFLite exporter:
```python
model.export(format='tflite', half=True, imgsz=640)
```
- **Half Precision (FP16)** reduces the model size from ~12MB down to ~6MB with zero drop in practical detection accuracy.
- Produces `ppe_model.tflite`.

## 5. Download Steps
In Colab, running cell 6 triggers the automatic download:
```python
from google.colab import files
files.download('ppe_model.tflite')
```
Place the downloaded file into:
```
CS21_student_[NAME]/models/ppe_model.tflite
```
*(Alternatively, run `python src/generate_test_model.py` to create a calibrated local test model immediately!)*

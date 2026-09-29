# CS21: Quantitative Performance Benchmarks

### Table 1: End-to-End Edge Analytics Performance Profile

| Metric                      | Value      | Reference                         |
|:----------------------------|:-----------|:----------------------------------|
| Edge In-Memory Latency      | 55.91 ms   | Tested across 100 frames          |
| Edge Real-Time Throughput   | 17.9 FPS   | 30 FPS real-time site requirement |
| Simulated Cloud Latency     | 1152.24 ms | 4G/LTE round-trip + cloud queue   |
| Edge Latency Speedup        | 20.6x      | Edge vs Cloud response speed      |
| Model Macro Precision       | 97.07%     | 5 classes evaluated               |
| Model Macro Recall          | 97.07%     | 5 classes evaluated               |
| Model Macro F1-Score        | 97.07%     | Harmonic mean of P and R          |
| Model mAP@0.50              | 94.20%     | IoU threshold 0.50                |
| Debounced False Alarm Rate  | 98.0%      | NORMAL scenario (N>=5 debounce)   |
| Raw Single-Frame Noise Rate | 100.0%     | Without consecutive debouncing    |
| Abnormal Detection Delay    | 0.0 ms     | 5 frames at 30 FPS                |
| Critical Detection Delay    | 0.0 ms     | 2 frames at 30 FPS                |
| Bandwidth Reduction         | 99.908%    | Raw video vs metadata alerts      |
| Data Compression Ratio      | 1090.7x    | Edge telemetry compaction         |

### Table 2: Per-Class Precision, Recall, and F1 Breakdown

| Class     |   precision |   recall |     f1 |   tp |   fp |   fn |
|:----------|------------:|---------:|-------:|-----:|-----:|-----:|
| person    |      1      |   0.9833 | 0.9916 |  295 |    0 |    5 |
| helmet    |      0.9756 |   0.9756 | 0.9756 |  240 |    6 |    6 |
| no_helmet |      0.9467 |   0.9595 | 0.953  |  142 |    8 |    6 |
| vest      |      0.9671 |   0.9671 | 0.9671 |  235 |    8 |    8 |
| no_vest   |      0.9367 |   0.9548 | 0.9457 |  148 |   10 |    7 |

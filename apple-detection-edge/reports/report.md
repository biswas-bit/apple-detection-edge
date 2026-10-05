# Models Deployment Metrics

## 1. Model Size and Computational Complexity

To evaluate the deployment efficiency of the trained models, their **trainable parameter count, model size, and computational complexity (GFLOPS)** were measured. These metrics provide an overview of the memory requirements and computational resources needed for model inference.

| Model          | Trainable Parameters | Model Size (MB) | GFLOPS |
| :------------- | -------------------: | --------------: | -----: |
| **Mine-Apple** |            3,011,043 |            5.92 |  8.200 |
| **Fuji**       |            3,011,043 |            5.93 |  8.200 |
| **Merged**     |            3,012,240 |            3.15 |  2.048 |

## 2. Detection Accuracy

| Model | mAP@0.5 | mAP@0.5:0.95 |
|:------|--------:|-------------:|
| **Mine-Apple** |0.65 |0.26 |
| **Fuji** | 0.83 |0.60 |
| **Merged** |0.69 | 0.31 |
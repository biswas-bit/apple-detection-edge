# Models Deployment Metrics

## 1. Model Size and Computational Complexity

To evaluate the deployment efficiency of the trained models, their **trainable parameter count, model size, and computational complexity (GFLOPS)** were measured. These metrics provide an overview of the memory requirements and computational resources needed for model inference.

| Model          | Trainable Parameters | Model Size (MB) | GFLOPS |
| :------------- | -------------------: | --------------: | -----: |
| **Mine-Apple** |            3,011,043 |            5.92 |  8.200 |
| **Fuji**       |            3,011,043 |            5.93 |  8.200 |
| **Merged**     |            3,012,240 |            3.15 |  2.048 |

## 2. Detection Accuracy

| **Model**  | **mAP@0.5** | **mAP@0.5:0.95** | **Precision** | **Recall** | **F1-score** |
| :--------- | ----------: | ---------------: | ------------: | ---------: | -----------: |
| Mine-Apple |        0.65 |             0.26 |             — |          — |            — |
| Fuji       |    **0.83** |         **0.60** |             — |          — |            — |
| Merged     |        0.69 |             0.31 |      **0.78** |   **0.62** |   **0.6953** |

## 3. Inference Latency 

| **Model**  | **Pre-processing Latency (ms)** | **Inference Latency (ms)** | **Post-processing Latency (ms)** | **End-to-End Latency (ms)** |   **FPS** |
| :--------- | ------------------------------: | -------------------------: | -------------------------------: | --------------------------: | --------: |
| Mine-Apple |                            0.88 |                      52.47 |                             1.29 |                       55.02 |     18.18 |
| Fuji       |                        **1.06** |                  **64.87** |                             1.58 |                       67.96 |     14.12 |
| Merged     |                            1.10 |                      34.51 |                         **1.63** |                   **37.81** | **26.45** |

## 3. RAM Usage

| **Model**  | **Format**     | **Model Load (MB)** | **Inference Overhead (MB)** | **Total Memory (MB)** |
| :--------- | :------------- | ------------------: | --------------------------: | --------------------: |
| Mine-Apple | FP32 (.pt)     |               19.78 |                       28.07 |                 47.86 |
| Fuji       | **FP32 (.pt)** |            **8.58** |                        8.01 |                 16.60 |
| Merged     | INT8 (TFLite)  |                0.02 |                    **4.55** |              **4.58** |

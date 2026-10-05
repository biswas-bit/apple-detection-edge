# Models Deployment Metrics

## 1. Model Size and Computational Complexity

To evaluate the deployment efficiency of the trained models, their **trainable parameter count, model size, and computational complexity (GFLOPS)** were measured. These metrics provide an overview of the memory requirements and computational resources needed for model inference.

| Model          | Trainable Parameters | Model Size (MB) | GFLOPS |
| :------------- | -------------------: | --------------: | -----: |
| **Mine-Apple** |            3,011,043 |            5.92 |  8.200 |
| **Fuji**       |            3,011,043 |            5.93 |  8.200 |
| **Merged**     |            3,012,240 |            3.15 |  2.048 |

### 1.1 Summary of Deployment Characteristics

The **Mine-Apple** and **Fuji** models have an almost identical number of trainable parameters and computational requirements, with model sizes of **5.92 MB** and **5.93 MB**, respectively. Both models require approximately **8.2 GFLOPS**, indicating similar computational complexity during inference.

In comparison, the **Merged** model contains slightly more trainable parameters (**3,012,240**) but has a considerably smaller model size of **3.15 MB**. It also requires only **2.048 GFLOPS**, representing substantially lower computational complexity than the individual models.

Overall, the deployment metrics suggest that the **Merged model provides a more computationally efficient configuration**, requiring less storage and significantly fewer floating-point operations while maintaining a comparable number of trainable parameters.

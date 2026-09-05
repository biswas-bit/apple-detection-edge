# AppleVision-Edge

Real-time apple detection and counting system, optimized for edge deployment on Raspberry Pi — from dataset curation to a production-ready quantized model.

<img src="apple-detection-edge/testimg/Screenshot 2026-09-05 134153.png" alt="Alt Text" width="900" height="500" />

---

## Overview

AppleVision-Edge fine-tunes a COCO-pretrained YOLOv8 detector for single-class apple detection, with the training pipeline specifically designed to handle real-world conditions: occlusion, clustering, varying apple density, and multiple imaging domains (isolated apples, orchard trees, close-up shots). The final model is quantized (INT8) and exported to TFLite for real-time inference on resource-constrained edge hardware.

**Use cases explored in this project:**
- Apple counting from static images and video
- Cumulative counting on a conveyor belt (via object tracking / line-crossing logic)
- Per-frame detection for drone-based orchard monitoring and robotic-arm picking targeting

Dataset engineering (multi-source annotation conversion and merging) lives in a companion repository:[apple-dataset-pipeline](https://github.com/biswas-bit/apple-data-unification).

---

## Key Features

- Fine-tuned YOLOv8n detector, initialized from COCO-pretrained weights
- Training pipeline validated across multiple datasets (isolated-apple images, MinneApple orchard benchmark, Fuji apple dataset) to address occlusion and domain-gap failures found during development
- INT8-quantized TFLite export, benchmarked for Raspberry Pi CPU inference
- Multiple counting strategies: per-frame detection, ByteTrack-based cumulative tracking, and line-crossing counting for conveyor-belt scenarios
- Reproducible experiment log with comparative metrics (see [Results](#results))

---

## Results

| Experiment | Dataset | Image size | mAP@0.5 | mAP@0.5:0.95 | Notes |
|---|---|---|---|---|---|
| Baseline | Isolated-apple set | 320 | 0.993 | 0.959 | Investigated and attributed to dataset homogeneity / limited scene diversity; failed to generalize on occluded/crowded test images |
| MinneApple (CIoU, default) | MinneApple detection subset | 320 | 0.710 | 0.296 | Addressed occlusion via orchard-scene data |
| MinneApple (GIoU ablation) | MinneApple detection subset | 320 | 0.687 | 0.283 | CIoU outperforms GIoU for this task |
| Fuji |  Fuji-SFM | 320 | 0.83 | 0.60 | much better detection when conf is set to 0.4 and IOU to 0.3 |



---

## Datasets

| Dataset | Description | Annotation format (original) |
|---|---|---|
| Isolated-apple set | Clean, mostly single/few-apple images | YOLO bounding box |
| [MinneApple](https://conservancy.umn.edu/items/e1bb4015-e92a-4295-822c-d21d277ecfbd) | Orchard tree images with heavy occlusion and clustering | Instance segmentation masks |
| Fuji apple dataset | Close-up apple imagery | VIA polygon CSV |

All annotations were converted to a unified YOLO bounding-box format. See [apple-dataset-pipeline](#) for the full conversion pipeline.

**Citation:**
N. Häni and P. Roy, "MinneApple: A Benchmark Dataset for Apple Detection and Segmentation," arXiv:1909.06441, 2019.

---

## Project Structure

```
apple-detection-edge/
├── README.md
├── data/
│   └── data.yaml
├── notebooks/                 # training/experiment notebooks
├── src/
│   ├── train.py
│   ├── export.py               # TFLite / INT8 export
│   └── inference.py             # image, video, and live-camera inference
├── models/                      # exported .tflite weights
├── benchmarks/                    # latency/accuracy tables (FP32 vs INT8, Pi vs Colab)
├── reports/                        # written experiment reports
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/biswas-bit/apple-detection-edge.git
cd apple-detection-edge
pip install -r requirements.txt
```

## Usage

**Train:**
```bash
python src/train.py --data data/data.yaml --epochs 150 --imgsz 320
```

**Export to edge-optimized TFLite:**
```bash
python src/export.py --weights runs/train/.../best.pt --data data/data.yaml
```

**Run inference:**
```bash
# Single image
python src/inference.py --weights models/best_int8.tflite --source test.jpg

# Live camera (cumulative tracking)
python src/inference.py --weights models/best_int8.tflite --source 0 --track
```

---

## Edge Deployment

- **Target hardware:** Raspberry Pi (generic ARM, CPU inference)
- **Export format:** TFLite, INT8 post-training quantization
- **Inference runtime:** `tflite-runtime` with XNNPACK delegate for accelerated CPU inference
- **Input resolution:** 320×320 (accuracy/latency trade-off point for Pi CPU)

*(Add measured latency/FPS numbers here once Pi benchmarking is complete.)*

---

## Limitations

- Detection precision (mAP@0.5:0.95) is lower in dense/occluded scenes, which may affect use cases requiring precise localization (e.g., robotic picking)
- Trained and validated on available public datasets; performance on entirely novel imaging conditions (e.g., different apple varieties, lighting) is not guaranteed
- Depth/distance estimation (needed for actual robotic grasping) is out of scope — this project addresses 2D detection only

## Future Work

- Higher-resolution training pass for improved box precision
- Expanded dataset coverage for drone/aerial and packing-line imaging conditions
- On-device latency benchmarking on physical Raspberry Pi hardware
- Depth estimation integration for robotic picking applications

---

## References

1. Häni, N. & Roy, P. "MinneApple: A Benchmark Dataset for Apple Detection and Segmentation." arXiv:1909.06441, 2019.
2. Ultralytics YOLOv8 Documentation. https://docs.ultralytics.com

## License

*(Add your chosen license — e.g., MIT — and note MinneApple's CC BY-NC-SA 3.0 license applies to that portion of the training data.)*

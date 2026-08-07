"""
export.py — Export a trained YOLOv8 apple-detection model to TFLite (INT8 quantized)
for edge deployment on Raspberry Pi / ARM devices.

Usage:
    python src/export.py --weights runs/train/yolov8n_apple/weights/best.pt --data data/data.yaml
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLOv8 model to TFLite INT8 for edge deployment")
    parser.add_argument("--weights", type=str, required=True,
                         help="Path to trained .pt weights (e.g. runs/train/.../best.pt)")
    parser.add_argument("--data", type=str, default="data/data.yaml",
                         help="data.yaml — used to build the INT8 calibration/representative dataset")
    parser.add_argument("--imgsz", type=int, default=320,
                         help="Export image size — must match the size used at inference time")
    parser.add_argument("--format", type=str, default="tflite",
                         choices=["tflite", "onnx", "edgetpu"],
                         help="Export format")
    parser.add_argument("--int8", action="store_true", default=True,
                         help="Enable INT8 quantization (recommended for Pi CPU)")
    parser.add_argument("--no-int8", dest="int8", action="store_false",
                         help="Disable INT8, export FP32 instead")
    return parser.parse_args()


def main():
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found at {weights_path}")

    model = YOLO(str(weights_path))

    export_path = model.export(
        format=args.format,
        int8=args.int8,
        imgsz=args.imgsz,
        data=args.data if args.int8 else None,  # calibration data only needed for INT8
    )

    print(f"\nExported model saved to: {export_path}")

    # Report file size — useful sanity check and a number worth recording in your report
    exported_file = Path(export_path)
    if exported_file.exists():
        size_mb = exported_file.stat().st_size / (1024 * 1024)
        print(f"Model size: {size_mb:.2f} MB")
    elif exported_file.is_dir():
        # Some export formats (e.g. saved_model) produce a directory
        total_size = sum(f.stat().st_size for f in exported_file.rglob("*") if f.is_file())
        print(f"Export directory size: {total_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
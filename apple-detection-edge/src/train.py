"""
train.py — Fine-tune a COCO-pretrained YOLOv8 model on the apple detection dataset.

Usage:
    python src/train.py --data data/data.yaml --epochs 100 --imgsz 320
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for apple detection")
    parser.add_argument("--data", type=str, default="data/data.yaml",
                         help="Path to data.yaml")
    parser.add_argument("--weights", type=str, default="yolov8n.pt",
                         help="Initial weights (COCO-pretrained)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=320,
                         help="Training image size — match your edge deployment target")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20,
                         help="Early stopping patience")
    parser.add_argument("--freeze", type=int, default=0,
                         help="Number of backbone layers to freeze (useful for small datasets)")
    parser.add_argument("--device", type=str, default="0",
                         help="'0' for GPU, 'cpu' for CPU")
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--name", type=str, default="yolov8n_apple")
    return parser.parse_args()


def main():
    args = parse_args()

    if not Path(args.data).exists():
        raise FileNotFoundError(f"data.yaml not found at {args.data}")

    model = YOLO(args.weights)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        freeze=args.freeze if args.freeze > 0 else None,
        device=args.device,
        project=args.project,
        name=args.name,
    )

    
    metrics = model.val()
    print(f"\nFinal mAP@0.5: {metrics.box.map50:.4f}")
    print(f"Final mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"\nBest weights saved to: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
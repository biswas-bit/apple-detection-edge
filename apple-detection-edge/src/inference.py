"""
inference.py — Quick, conventional apple counting test (no CLI args).
Just edit the variables below and run. Will be refactored into a reusable
CLI tool later once the core logic is confirmed working.
"""

import cv2
from ultralytics import YOLO

# ---- Edit these directly for now ----
WEIGHTS_PATH = "models/best_int8.tflite"
IMAGE_PATH = "testimg/test2.jpg"
IMG_SIZE = 320
CONF_THRESHOLD = 0.25
SAVE_PATH = "annotedimg/annotated_result2.jpg"
# --------------------------------------

model = YOLO(WEIGHTS_PATH)

results = model(IMAGE_PATH, imgsz=IMG_SIZE, conf=CONF_THRESHOLD)
r = results[0]

count = len(r.boxes)
print(f"Apples detected: {count}")

annotated = r.plot()
cv2.imwrite(SAVE_PATH, annotated)
print(f"Annotated image saved to {SAVE_PATH}")
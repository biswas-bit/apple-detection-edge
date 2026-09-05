"""
inference_tree_detection.py — Detect apples on trees for drone/robotic-arm targeting.
Outputs bounding boxes + centroids per frame (no counting/tracking needed).
"""

import os
import cv2
from ultralytics import YOLO

WEIGHTS_PATH = "apple-detection-edge/models/appledetection_fuji_and_mineapple.tflite"
VIDEO_PATH = "apple-detection-edge/testimg/apple_in_trees.mp4"
IMG_SIZE = 320
CONF_THRESHOLD = 0.4    
IOU_THRESHOLD = 0.3
SAVE_DIR = "apple-detection-edge/annotedimg"
SAVE_PATH = os.path.join(SAVE_DIR, "tree_detection.mp4")

os.makedirs(SAVE_DIR, exist_ok=True)

model = YOLO(WEIGHTS_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 20
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(SAVE_PATH, fourcc, fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
    r = results[0]

    annotated = r.plot()
    cv2.putText(annotated, f"Apples detected: {len(r.boxes)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Extract centroid of each detected apple — this is what a drone/arm would target
    for box in r.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)  # mark centroid

    writer.write(annotated)
    cv2.imshow("Apple Detector", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

print(f"Annotated video saved to: {SAVE_PATH}")
"""
inference_video.py — Detect apples in a video file (frame by frame).
"""

import cv2
from ultralytics import YOLO

WEIGHTS_PATH = "apple-detection-edge/models/best.pt"
VIDEO_PATH = "apple-detection-edge/testimg/Apple for food belt conveyor.mp4"       # <-- your video file
IMG_SIZE = 320
CONF_THRESHOLD = 0.1
SAVE_PATH = "apple-detection-edge/annotedimg"  
IOU_THRESHOLD = 0.3

model = YOLO(WEIGHTS_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)   # file path instead of camera index 0

# Set up video writer to save annotated output
writer = None
if SAVE_PATH:
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(SAVE_PATH, fourcc, fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break   # end of video

    results = model(frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, iou = IOU_THRESHOLD)
    r = results[0]
    annotated = r.plot()

    cv2.putText(annotated, f"Count: {len(r.boxes)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if writer:
        writer.write(annotated)

    cv2.imshow("Apple Counter", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()
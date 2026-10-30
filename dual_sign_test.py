
#!/usr/bin/env python3
"""GO + STOP live camera test. No motor, UART, or ROS commands."""

import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from ugv_camera.core import CameraConfig
from ugv_camera.capture import Capture
from ugv_camera.outputs import StreamServer

# --------------------------------------------------
# MODEL PATHS
# --------------------------------------------------

GO_PATH = "/home/sdp2026/Desktop/jetson_camera_go/go_sign_v3_low_lr.pt"
STOP_PATH = "/home/sdp2026/YOLO_scripts/yolov8n.pt"

# Detection confidence thresholds
GO_THRESHOLD = 0.25
STOP_THRESHOLD = 0.10

# COCO class number for STOP signs
STOP_CLASS = 11


# --------------------------------------------------
# DRAW DETECTION BOXES
# --------------------------------------------------

def draw_boxes(frame, result, name, color):
    for box in result.boxes:
        x1, y1, x2, y2 = map(
            int, box.xyxy[0].tolist()
        )

        confidence = float(box.conf[0])

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            frame,
            f"{name} {confidence:.2f}",
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

def main():

    os.environ["YOLO_AUTOINSTALL"] = "False"

    # Verify models exist
    for path in [GO_PATH, STOP_PATH]:
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    if not torch.cuda.is_available():
        raise RuntimeError("Jetson CUDA is not available")

    camera_env = {
        "CUDA_VISIBLE_DEVICES": os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        )
    }

    # --------------------------------------------------
    # LOAD BOTH YOLO MODELS
    # --------------------------------------------------

    print("Loading GO V3 model...")
    go_model = YOLO(GO_PATH, task="detect")

    print("Loading original YOLOv8 STOP model...")
    stop_model = YOLO(STOP_PATH, task="detect")

    # Warm up models
    dummy = np.zeros(
        (640, 640, 3),
        dtype=np.uint8
    )

    go_model.predict(
        dummy,
        device=0,
        verbose=False
    )

    stop_model.predict(
        dummy,
        device=0,
        verbose=False
    )

    stream = None
    capture = None

    try:

        # --------------------------------------------------
        # START CAMERA STREAM
        # --------------------------------------------------

        stream = StreamServer(
            "127.0.0.1",
            8080
        )

        capture = Capture(
            CameraConfig(),
            environment=camera_env
        )

        print("GO + STOP detection running")
        print("View: http://127.0.0.1:8080/video")
        print("VISUAL TEST ONLY - NO MOTOR COMMANDS")

        # --------------------------------------------------
        # LIVE DETECTION LOOP
        # --------------------------------------------------

        while True:

            _, timestamp, frame = capture.read(10)

            # -----------------------------
            # GO SIGN DETECTION
            # -----------------------------

            go = go_model.predict(
                frame,
                conf=0.10,
                imgsz=640,
                device=0,
                verbose=False
            )[0]

            # -----------------------------
            # STOP SIGN DETECTION
            # -----------------------------

            # Run original YOLOv8 with all classes.
            # Then select only STOP sign detections.

            stop = stop_model.predict(
                frame,
                conf=0.10,
                imgsz=640,
                device=0,
                verbose=False
            )[0]

            # COCO class 11 = STOP sign
            stop.boxes = stop.boxes[
                stop.boxes.cls == STOP_CLASS
            ]

            # -----------------------------
            # GET CONFIDENCE SCORES
            # -----------------------------

            go_conf = max(
                (
                    float(b.conf[0])
                    for b in go.boxes
                ),
                default=0.0
            )

            stop_conf = max(
                (
                    float(b.conf[0])
                    for b in stop.boxes
                ),
                default=0.0
            )

            go_seen = go_conf >= GO_THRESHOLD
            stop_seen = stop_conf >= STOP_THRESHOLD

            # -----------------------------
            # DECISION LOGIC
            # -----------------------------

            if go_seen and stop_seen:
                state = "HOLD - CONFLICT"

            elif stop_seen:
                state = "STOP DETECTED"

            elif go_seen:
                state = "GO CANDIDATE"

            else:
                state = "HOLD - NO SIGN"

            # -----------------------------
            # ANNOTATE CAMERA FRAME
            # -----------------------------

            annotated = frame.copy()

            # Green boxes = GO
            draw_boxes(
                annotated,
                go,
                "GO",
                (0, 255, 0)
            )

            # Pink boxes = STOP
            draw_boxes(
                annotated,
                stop,
                "STOP",
                (255, 0, 255)
            )

            # Black status banner
            cv2.rectangle(
                annotated,
                (0, 0),
                (850, 95),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                annotated,
                state,
                (15, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

            cv2.putText(
                annotated,
                f"GO: {go_conf:.2f}   "
                f"STOP: {stop_conf:.2f}   "
                f"| NO MOTOR CONTROL",
                (15, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            # -----------------------------
            # DISCARD OLD FRAMES
            # -----------------------------

            if (
                time.monotonic_ns() - timestamp
            ) > 2e9:
                continue

            # -----------------------------
            # SEND IMAGE TO BROWSER
            # -----------------------------

            ok, jpeg = cv2.imencode(
                ".jpg",
                annotated
            )

            if ok:
                stream.hub.publish(
                    jpeg.tobytes(),
                    timestamp / 1e9
                )

    except KeyboardInterrupt:
        print("Camera test stopped")

    finally:

        if capture is not None:
            capture.close()

        if stream is not None:
            stream.close()


if __name__ == "__main__":
    main()

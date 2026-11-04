
#!/usr/bin/env python3
"""
GO V4 + STOP detection with color verification.
VISUAL TEST ONLY. No ROS, UART, or motor commands.
"""

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
# MODELS
# --------------------------------------------------

GO_PATH = (
    "/home/sdp2026/Desktop/jetson_camera_toolkit/"
    "models/go_sign_v4_paper_phone.pt"
)

STOP_PATH = "/home/sdp2026/YOLO_scripts/yolov8n.pt"

STOP_CLASS = 11

GO_THRESHOLD = 0.25
STOP_THRESHOLD = 0.25

# Initial color-verification settings.
# These can be adjusted using live results.
MIN_COLOR_FRACTION = 0.16
COLOR_DOMINANCE = 2.0

# Require consecutive matching frames.
REQUIRED_FRAMES = 3


# --------------------------------------------------
# COLOR VERIFICATION
# --------------------------------------------------

def get_color_fractions(frame, box):
    """
    Measure the green and red fractions inside a
    detection box, excluding most of its border.

    OpenCV hue range: 0-179.
    """

    h, w = frame.shape[:2]

    x1, y1, x2, y2 = map(
        int, box.xyxy[0].tolist()
    )

    # Move inward to reduce background and
    # white paper included in bounding boxes.
    dx = int((x2 - x1) * 0.12)
    dy = int((y2 - y1) * 0.12)

    x1 = max(0, x1 + dx)
    y1 = max(0, y1 + dy)
    x2 = min(w, x2 - dx)
    y2 = min(h, y2 - dy)

    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0

    crop = frame[y1:y2, x1:x2]

    hsv = cv2.cvtColor(
        crop, cv2.COLOR_BGR2HSV
    )

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Reject unsaturated gray and white areas.
    colorful = (
        (saturation >= 40)
        & (value >= 35)
    )

    # Include the teal-green appearance seen
    # from the Jetson under lab lighting.
    green_mask = (
        (hue >= 35)
        & (hue <= 105)
        & colorful
    )

    # Red wraps around both ends of HSV hue.
    red_mask = (
        ((hue <= 22) | (hue >= 165))
        & colorful
    )

    total = hue.size

    green_fraction = (
        np.count_nonzero(green_mask) / total
    )

    red_fraction = (
        np.count_nonzero(red_mask) / total
    )

    return float(green_fraction), float(red_fraction)


def verify_color(green_fraction, red_fraction, expected):
    """
    Require enough of the expected color and
    make sure it dominates the opposite color.
    """

    if expected == "GO":
        return (
            green_fraction >= MIN_COLOR_FRACTION
            and green_fraction >= (
                COLOR_DOMINANCE * red_fraction
            )
        )

    if expected == "STOP":
        return (
            red_fraction >= MIN_COLOR_FRACTION
            and red_fraction >= (
                COLOR_DOMINANCE * green_fraction
            )
        )

    return False


# --------------------------------------------------
# DETECTION FILTERING
# --------------------------------------------------

def filter_detections(frame, result, name, threshold):
    """
    Only retain YOLO predictions that pass
    confidence AND color verification.
    """

    accepted = []
    raw_confidence = 0.0
    raw_green = 0.0
    raw_red = 0.0

    for box in result.boxes:

        confidence = float(box.conf[0])

        green, red = get_color_fractions(
            frame, box
        )

        if confidence > raw_confidence:
            raw_confidence = confidence
            raw_green = green
            raw_red = red

        if confidence < threshold:
            continue

        if not verify_color(green, red, name):
            continue

        accepted.append(
            (box, confidence, green, red)
        )

    return (
        accepted,
        raw_confidence,
        raw_green,
        raw_red
    )


# --------------------------------------------------
# DRAW VERIFIED DETECTIONS
# --------------------------------------------------

def draw_verified(frame, detections, name, color):

    for box, confidence, green, red in detections:

        x1, y1, x2, y2 = map(
            int, box.xyxy[0].tolist()
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        cv2.putText(
            frame,
            f"{name} {confidence:.2f}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2
        )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    os.environ["YOLO_AUTOINSTALL"] = "False"

    for path in (GO_PATH, STOP_PATH):
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Jetson CUDA is unavailable"
        )

    print("Loading GO V4 model...")
    go_model = YOLO(GO_PATH, task="detect")

    print("Loading YOLOv8 STOP model...")
    stop_model = YOLO(STOP_PATH, task="detect")

    # Warm up models before starting camera.
    dummy = np.zeros(
        (640, 640, 3), dtype=np.uint8
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

    previous_state = None
    consecutive_frames = 0

    try:
        stream = StreamServer(
            "127.0.0.1", 8080
        )

        capture = Capture(
            CameraConfig(),
            environment={
                "CUDA_VISIBLE_DEVICES":
                    os.environ.get("CUDA_VISIBLE_DEVICES")
            }
        )

        print("GO V4 + STOP color verification running")
        print("http://127.0.0.1:8080/video")
        print("VISUAL TEST ONLY - NO MOTOR COMMANDS")

        while True:

            _, timestamp, frame = capture.read(10)

            # ------------------------------------------
            # YOLO GO V4
            # ------------------------------------------

            go_result = go_model.predict(
                frame,
                conf=0.10,
                imgsz=640,
                device=0,
                verbose=False
            )[0]

            # ------------------------------------------
            # YOLO STOP - COCO CLASS 11
            # ------------------------------------------

            stop_result = stop_model.predict(
                frame,
                conf=0.10,
                classes=[STOP_CLASS],
                imgsz=640,
                device=0,
                verbose=False
            )[0]

            # ------------------------------------------
            # COLOR VALIDATION
            # ------------------------------------------

            (
                go_valid,
                go_raw,
                go_green,
                go_red
            ) = filter_detections(
                frame,
                go_result,
                "GO",
                GO_THRESHOLD
            )

            (
                stop_valid,
                stop_raw,
                stop_green,
                stop_red
            ) = filter_detections(
                frame,
                stop_result,
                "STOP",
                STOP_THRESHOLD
            )

            go_conf = max(
                (item[1] for item in go_valid),
                default=0.0
            )

            stop_conf = max(
                (item[1] for item in stop_valid),
                default=0.0
            )

            # ------------------------------------------
            # DECISION
            # ------------------------------------------

            if go_valid and stop_valid:
                candidate = "HOLD - BOTH SIGNS"

            elif stop_valid:
                candidate = "STOP"

            elif go_valid:
                candidate = "GO"

            else:
                candidate = "HOLD - NO VERIFIED SIGN"

            # Require consistent observations.
            if candidate == previous_state:
                consecutive_frames += 1
            else:
                previous_state = candidate
                consecutive_frames = 1

            if candidate == "GO":
                if consecutive_frames >= REQUIRED_FRAMES:
                    state = "GO VERIFIED"
                else:
                    state = "HOLD - VERIFYING GO"

            elif candidate == "STOP":
                if consecutive_frames >= REQUIRED_FRAMES:
                    state = "STOP VERIFIED"
                else:
                    state = "HOLD - VERIFYING STOP"

            else:
                state = candidate

            # ------------------------------------------
            # ANNOTATE VIDEO
            # ------------------------------------------

            annotated = frame.copy()

            draw_verified(
                annotated,
                go_valid,
                "GO",
                (0, 255, 0)
            )

            draw_verified(
                annotated,
                stop_valid,
                "STOP",
                (255, 0, 255)
            )

            height, width = annotated.shape[:2]

            cv2.rectangle(
                annotated,
                (0, 0),
                (min(width - 1, 1200), 133),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                annotated,
                state,
                (15, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2
            )

            cv2.putText(
                annotated,
                f"Verified GO: {go_conf:.2f}  "
                f"STOP: {stop_conf:.2f}  "
                "| NO MOTOR CONTROL",
                (15, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.64,
                (255, 255, 255),
                2
            )

            # Diagnostic values to calibrate color.
            cv2.putText(
                annotated,
                f"Raw GO:{go_raw:.2f} "
                f"(green:{go_green:.2f} red:{go_red:.2f})  "
                f"Raw STOP:{stop_raw:.2f} "
                f"(red:{stop_red:.2f} green:{stop_green:.2f})",
                (15, 107),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.51,
                (220, 220, 220),
                2
            )

            # ------------------------------------------
            # STREAM
            # ------------------------------------------

            if (
                time.monotonic_ns() - timestamp
            ) > 2_000_000_000:
                continue

            ok, jpeg = cv2.imencode(
                ".jpg",
                annotated,
                [cv2.IMWRITE_JPEG_QUALITY, 83]
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

#!/usr/bin/env python3
"""Read-only Jetson CSI camera test of the offline white-tape lane tracker.

Uses the known working ugv_camera camera/stream components. No motor,
UART, ROS, autonomous commands, or YOLO inference. Safe for stationary tests.

Run from inside the jetson_camera_toolkit directory:
    python3 lane_live_test.py

Browser via existing SSH forwarding:
    http://127.0.0.1:8081/video
"""

import argparse
import os
import time

import cv2

from lane_offline import process
from ugv_camera.core import CameraConfig
from ugv_camera.capture import Capture
from ugv_camera.outputs import StreamServer


def main():
    parser = argparse.ArgumentParser(description="Visual-only lane tracker")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--show-mask", action="store_true",
                        help="Show white-tape candidate mask alongside the camera")
    args = parser.parse_args()

    stream = None
    camera = None
    prev_time = None
    n = 0
    try:
        # Matches the camera and streaming interfaces already verified
        # by the dual_sign_test.py script on this Jetson.
        stream = StreamServer("127.0.0.1", args.port)
        camera = Capture(
            CameraConfig(),
            environment={"CUDA_VISIBLE_DEVICES":
                         os.environ.get("CUDA_VISIBLE_DEVICES")}
        )

        print("LANE TRACKING - VISUAL TEST ONLY")
        print("NO MOTOR, UART OR ROS COMMANDS")
        print(f"Camera stream at 127.0.0.1:{args.port}/video")
        print("Keep the vehicle stationary; move the camera or tape to test.")

        while True:
            _, timestamp, frame = camera.read(10)
            if frame is None:
                continue

            # The offline prototype was calibrated against 1280x720
            # photos. Preserve aspect ratio; use exactly that resolution.
            test_frame = cv2.resize(frame, (1280, 720),
                                    interpolation=cv2.INTER_AREA)
            overlay, mask, result = process(test_frame)

            # Show mask separately when requested for debugging.
            if args.show_mask:
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                overlay = cv2.hconcat([overlay, mask_bgr])

            now = time.monotonic()
            fps = 0.0 if prev_time is None else 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(overlay, f"Processing FPS: {fps:.1f}",
                        (18, overlay.shape[0] - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(overlay, f"Processing FPS: {fps:.1f}",
                        (18, overlay.shape[0] - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            ok, jpeg = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 83])
            if ok:
                stream.hub.publish(jpeg.tobytes(), timestamp / 1e9)

            n += 1
            if n % 30 == 0:
                print("Lane:", result["direction"],
                      "offset:", result["error"],
                      "floor:", result["floor_brightness"],
                      "fps:", round(fps, 1), flush=True)

    except KeyboardInterrupt:
        print("Lane tracking stopped")
    finally:
        if camera is not None:
            camera.close()
        if stream is not None:
            stream.close()


if __name__ == "__main__":
    main()

"""Command line entry point; optional integrations are disabled by default."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import uuid

from .core import CameraConfig, detection, finite_positive, json_text, observation


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnose", action="store_true", help="Report libraries and CUDA without opening camera/UART/ROS")
    p.add_argument("--print-pipeline", action="store_true")
    p.add_argument("--model", default="yolov8n.pt", help="Existing trusted local model; no implicit download")
    p.add_argument("--device", choices=["0", "cpu"], default="0", help="CUDA required by default; CPU must be explicit")
    p.add_argument("--no-inference", action="store_true", help="Camera only, no model/CUDA import")
    p.add_argument("--track", action="store_true", help="ByteTrack persistent object IDs")
    p.add_argument("--classes", nargs="+", type=int)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--max-det", type=int, default=50)
    p.add_argument("--show", action="store_true", help="Opt-in desktop window; q exits")
    p.add_argument("--record", choices=["off", "auto", "mp4", "avi"], default="off")
    p.add_argument("--record-fps", type=finite_positive, default=10.0,
                   help="Playback FPS; one recorded frame per processed frame; timestamps in CSV")
    p.add_argument("--record-required", action="store_true", help="Fail session if recorder initialization fails")
    p.add_argument("--stream", action="store_true", help="MJPEG at /video and /snapshot.jpg")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--ros", action="store_true", help="Publish isolated /camera_observer observations")
    p.add_argument("--ros-images", action="store_true")
    p.add_argument("--uart-port", help="Dedicated telemetry receiver only; no default device")
    p.add_argument("--uart-observer-confirmed", action="store_true",
                   help="Confirm this port connects exclusively to an observation receiver, not motor hardware")
    p.add_argument("--uart-baud", type=int, default=115200)
    p.add_argument("--uart-rate", type=finite_positive, default=2.0)
    p.add_argument("--source-file", type=Path, help="Replay an existing video instead of CSI (paced, latest frame)")
    p.add_argument("--output", type=Path, default=Path("runs/camera"))
    p.add_argument("--frames", type=int, default=0, help="Stop after N processed frames; 0 unlimited")
    p.add_argument("--seconds", type=finite_positive, help="Maximum observation-loop duration")
    p.add_argument("--frame-timeout", type=finite_positive, default=10.0)
    p.add_argument("--max-frame-age", type=finite_positive, default=2.0,
                   help="Reject observations older than this after inference; does not measure sensor latency")
    p.add_argument("--min-free-mb", type=finite_positive, default=256.0)
    for name, default in [("sensor-id", 0), ("sensor-mode", 2), ("capture-width", 1920),
                          ("capture-height", 1080), ("capture-fps", 60), ("width", 1280),
                          ("height", 720), ("flip", 0)]:
        p.add_argument("--" + name, type=int, default=default)
    return p


def camera_config(a):
    return CameraConfig(a.sensor_id, a.sensor_mode, a.capture_width, a.capture_height,
                        a.capture_fps, a.width, a.height, a.flip)


def validate(a):
    camera_config(a).validate()
    if not 0 < a.conf <= 1 or not 32 <= a.imgsz <= 2048 or a.imgsz % 32:
        raise ValueError("Confidence must be (0,1]; imgsz must be a multiple of 32 in 32..2048")
    if a.frames < 0 or not 1 <= a.max_det <= 300 or not 1 <= a.port <= 65535:
        raise ValueError("Invalid frame count, detection limit, or port")
    if a.classes and any(c < 0 for c in a.classes):
        raise ValueError("Class IDs must be nonnegative")
    if a.track and a.no_inference:
        raise ValueError("Tracking requires inference")
    if a.ros_images and not a.ros:
        raise ValueError("--ros-images requires --ros")
    if a.record_required and a.record == "off":
        raise ValueError("--record-required needs a recording mode")
    if bool(a.uart_port) != bool(a.uart_observer_confirmed):
        raise ValueError("UART requires both an explicit dedicated port and --uart-observer-confirmed")
    if a.uart_rate > 5 or a.uart_baud < 115200:
        raise ValueError("Telemetry limited to <=5 Hz at >=115200 baud")
    if a.source_file and not a.source_file.is_file():
        raise ValueError("Replay file does not exist")
    if a.show and not os.environ.get("DISPLAY"):
        raise ValueError("--show requires a working desktop DISPLAY; use --stream over SSH")


def diagnose():
    import cv2
    import torch
    data = {"python": sys.version, "opencv": cv2.__version__, "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "video_build": [s.strip() for s in cv2.getBuildInformation().splitlines()
                            if any(k in s for k in ("GStreamer", "FFMPEG", "GUI:"))]}
    for name in ("ultralytics", "numpy", "torchvision", "pyserial"):
        try:
            data[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            data[name] = "not installed"
    print(json.dumps(data, indent=2))
    return 0 if data["cuda_available"] else 2


def load_model(a):
    if a.no_inference:
        return None
    model_path = Path(a.model).expanduser().resolve()
    if not model_path.is_file():
        raise ValueError(f"Provide an existing trusted local model with --model: {model_path}")
    # Ultralytics otherwise may install missing tracking packages during execution.
    os.environ["YOLO_AUTOINSTALL"] = "False"
    import torch
    from ultralytics import YOLO
    if a.device == "0" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Preserve NVIDIA PyTorch; diagnose before running. CPU needs --device cpu.")
    model = YOLO(str(model_path), task="detect")
    if a.track:
        try:
            import lap  # noqa: F401; fail before opening camera if dependency is absent
        except ImportError as exc:
            raise RuntimeError("Tracking requires lap>=0.5.12; see README's isolated tracking setup") from exc
    # Warm up before camera capture, so model initialization does not age observations.
    import numpy as np
    model.predict(np.zeros((a.imgsz, a.imgsz, 3), dtype=np.uint8), device=a.device,
                  imgsz=a.imgsz, conf=a.conf, verbose=False)
    return model


def infer(model, frame, a):
    if model is None:
        return None, []
    kwargs = dict(source=frame, device=a.device, imgsz=a.imgsz, conf=a.conf,
                  classes=a.classes, max_det=a.max_det, verbose=False)
    result = (model.track(**kwargs, persist=True, tracker="bytetrack.yaml") if a.track
              else model.predict(**kwargs))[0]
    h, w = frame.shape[:2]
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            track_id = int(box.id[0].item()) if box.id is not None else None
            detections.append(detection(class_id, result.names[class_id], box.conf[0].item(),
                                        box.xyxy[0].tolist(), w, h, track_id))
    return result, detections


def run(a):
    import cv2
    from .capture import Capture
    from .outputs import Recorder, RosObserver, StreamServer, UartObserver

    # Ultralytics changes CUDA_VISIBLE_DEVICES (including hiding all GPUs for
    # CPU inference). Argus still needs CUDA in the separate capture worker.
    camera_environment = {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
    model = load_model(a)
    session_id = uuid.uuid4().hex
    directory = a.output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + session_id[:8])
    directory.mkdir(parents=True, exist_ok=False)
    metadata = {"session_id": session_id, "arguments": vars(a),
                "pipeline": camera_config(a).pipeline() if not a.source_file else None,
                "opencv": cv2.__version__, "started_unix_ns": time.time_ns()}
    (directory / "session.json").write_text(json.dumps(metadata, default=str, indent=2) + "\n")
    print(f"Session: {directory.resolve()}", flush=True)
    count, state, error = 0, "stopped", None
    stopping = False
    old_handlers = {}

    def stop_requested(signum, frame):
        nonlocal stopping
        stopping = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, stop_requested)
    ros = None
    started = time.monotonic()
    last_log = started
    recorder = None
    recording_attempted = False
    try:
        with ExitStack() as stack:
            records = stack.enter_context((directory / "observations.jsonl").open("x"))
            stream = StreamServer(a.host, a.port, a.max_frame_age) if a.stream else None
            if stream:
                stack.callback(stream.close)
                print(f"Stream: http://{a.host}:{a.port}/video", flush=True)
            if a.ros:
                ros = RosObserver(a.ros_images)
                stack.callback(ros.close)
                stack.callback(lambda: ros.terminal_status("fault" if sys.exc_info()[0] else state))
            uart = UartObserver(a.uart_port, a.uart_baud, a.uart_rate) if a.uart_port else None
            if uart:
                stack.callback(uart.close)
            capture = Capture(camera_config(a), str(a.source_file) if a.source_file else None,
                              environment=camera_environment)
            stack.callback(capture.close)
            if a.show:
                stack.callback(cv2.destroyAllWindows)
            try:
                while not stopping and (not a.frames or count < a.frames):
                    if a.seconds and time.monotonic() - started >= a.seconds:
                        break
                    if shutil.disk_usage(directory).free < a.min_free_mb * 1024 * 1024:
                        raise RuntimeError("Output free-space reserve reached")
                    remaining = a.seconds - (time.monotonic() - started) if a.seconds else a.frame_timeout
                    try:
                        stamp, mono, frame = capture.read(min(a.frame_timeout, max(0.001, remaining)))
                    except EOFError:
                        state = "eof"
                        break
                    except TimeoutError:
                        if a.seconds and count and time.monotonic() - started >= a.seconds:
                            break
                        raise
                    result, boxes = infer(model, frame, a)
                    age = (time.monotonic_ns() - mono) / 1e9
                    if age > a.max_frame_age:
                        raise RuntimeError(f"Processed frame is stale ({age:.3f}s); observation suppressed")
                    h, w = frame.shape[:2]
                    obs = observation(session_id, count, stamp, mono, w, h, boxes, model is not None)
                    obs["processing_age_seconds"] = age
                    obs["published_unix_ns"] = time.time_ns()
                    need_visual = a.show or a.record != "off" or a.stream or a.ros_images
                    visual = result.plot() if result is not None and need_visual else frame
                    if a.record != "off" and not recording_attempted:
                        recording_attempted = True
                        try:
                            recorder = Recorder(directory, visual, a.record_fps, a.record)
                            stack.callback(recorder.close)
                            print(f"Recording: {recorder.path}", flush=True)
                        except Exception as exc:
                            (directory / "recording_error.txt").write_text(str(exc) + "\n")
                            if a.record_required:
                                raise
                            print(f"Recording disabled: {exc}", file=sys.stderr)
                    if recorder:
                        recorder.write(visual, obs)
                    jpeg = None
                    if stream or a.ros_images:
                        ok, encoded = cv2.imencode(".jpg", visual, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if not ok:
                            raise RuntimeError("JPEG encoding failed")
                        jpeg = encoded.tobytes()
                    # Serialization/plotting/encoding must not send an already stale observation.
                    if (time.monotonic_ns() - mono) / 1e9 > a.max_frame_age:
                        raise RuntimeError("Frame became stale while preparing outputs")
                    records.write(json_text(obs) + "\n")
                    records.flush()
                    if uart:
                        uart.publish(obs)
                    if ros:
                        ros.publish(obs, jpeg)
                    if stream:
                        stream.hub.publish(jpeg, mono / 1e9)
                    count += 1
                    now = time.monotonic()
                    if now - last_log >= 1:
                        print(f"Processed {count} | average {count / (now - started):.1f} FPS | detections {len(boxes)}", flush=True)
                        last_log = now
                    if a.show:
                        cv2.imshow("UGV camera observer", visual)
                        if cv2.waitKey(1) & 0xff == ord("q"):
                            break
            except BaseException:
                state = "fault"
                raise
    except KeyboardInterrupt:
        state = "interrupted"
    except Exception as exc:
        state, error = "fault", str(exc)
        print(f"ERROR: {error}", file=sys.stderr)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        summary = {"state": state, "error": error, "processed_frames": count,
                   "elapsed_seconds": time.monotonic() - started,
                   "recorded_frames": recorder.count if recorder else 0,
                   "recording": str(recorder.path) if recorder else None}
        (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    return 1 if state == "fault" else 0


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    try:
        validate(a)
        if a.print_pipeline:
            print(camera_config(a).pipeline())
            return 0
        if a.diagnose:
            return diagnose()
        return run(a)
    except (ValueError, RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

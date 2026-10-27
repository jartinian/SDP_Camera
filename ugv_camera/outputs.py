"""Optional recording, MJPEG, ROS observations, and explicitly enabled telemetry."""

import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time

from .core import json_text, uart_packet


class Recorder:
    """Software codecs only. Preflight write/decode before accepting a backend."""

    def __init__(self, directory, first_frame, fps=10.0, mode="auto"):
        import cv2
        self.cv2 = cv2
        self.writer = None
        self.csv_file = None
        self.count = 0
        self.shape = first_frame.shape
        h, w = self.shape[:2]
        candidates = [("mp4v", ".mp4"), ("MJPG", ".avi")]
        if mode != "auto":
            candidates = [candidates[0 if mode == "mp4" else 1]]
        errors = []
        for codec, suffix in candidates:
            probe = Path(directory) / ("codec_probe" + suffix)
            writer = cv2.VideoWriter(str(probe), cv2.CAP_FFMPEG,
                                     cv2.VideoWriter_fourcc(*codec), fps, (w, h))
            opened = writer.isOpened()
            if opened:
                writer.write(first_frame)
            writer.release()
            reader = cv2.VideoCapture(str(probe), cv2.CAP_FFMPEG)
            ok, decoded = reader.read()
            reader.release()
            probe.unlink(missing_ok=True)
            if not opened or not ok or decoded is None or decoded.shape[:2] != (h, w):
                errors.append(codec)
                continue
            self.path = Path(directory) / ("video" + suffix)
            self.writer = cv2.VideoWriter(str(self.path), cv2.CAP_FFMPEG,
                                         cv2.VideoWriter_fourcc(*codec), fps, (w, h))
            if self.writer.isOpened():
                break
            self.writer.release()
            self.writer = None
        if self.writer is None:
            raise RuntimeError(f"Software recorder unavailable (tried {errors}); use --record off")
        try:
            self.csv_file = (Path(directory) / "video_timestamps.csv").open("x", newline="")
            self.csv = csv.writer(self.csv_file)
            self.csv.writerow(["video_frame", "observation_sequence", "received_unix_ns", "received_monotonic_ns"])
        except BaseException:
            self.writer.release()
            raise

    def write(self, frame, obs):
        if frame.shape != self.shape:
            raise ValueError("Recording dimensions changed")
        self.writer.write(frame)
        self.csv.writerow([self.count, obs["sequence"], obs["capture_received_unix_ns"],
                           obs["capture_received_monotonic_ns"]])
        self.count += 1
        if self.count % 30 == 0:
            self.csv_file.flush()

    def close(self):
        self.writer.release()
        self.csv_file.close()
        # OpenCV write() has no success return. Decode after finalization to catch gross failures.
        reader = self.cv2.VideoCapture(str(self.path), self.cv2.CAP_FFMPEG)
        ok, _ = reader.read()
        reader.release()
        if self.count and not ok:
            raise RuntimeError(f"Recording failed final decode check: {self.path}")


class FrameHub:
    def __init__(self, stale_seconds=2.0):
        self.condition = threading.Condition()
        self.jpeg = None
        self.sequence = 0
        self.updated = 0.0
        self.closed = False
        self.stale_seconds = stale_seconds

    def publish(self, jpeg, received_monotonic=None):
        with self.condition:
            self.jpeg = jpeg
            self.sequence += 1
            self.updated = time.monotonic() if received_monotonic is None else received_monotonic
            self.condition.notify_all()

    def wait(self, previous):
        with self.condition:
            self.condition.wait_for(lambda: self.closed or self.sequence != previous,
                                    timeout=self.stale_seconds)
            if self.closed or self.jpeg is None or time.monotonic() - self.updated > self.stale_seconds:
                return None
            if self.sequence == previous:
                return None
            return self.sequence, self.jpeg

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


class StreamServer:
    def __init__(self, host, port, stale_seconds=2.0):
        self.hub = FrameHub(stale_seconds)
        hub = self.hub
        slots = threading.BoundedSemaphore(4)

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(3)

            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == "/":
                    body = b'<title>UGV camera observer</title><h1>Camera observations</h1><img src="/video">'
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path not in ("/video", "/snapshot.jpg"):
                    self.send_error(404)
                    return
                if not slots.acquire(blocking=False):
                    self.send_error(503, "Maximum four image clients")
                    return
                try:
                    item = hub.wait(-1)
                    if item is None:
                        self.send_error(503, "Camera image unavailable or stale")
                        return
                    self.send_response(200)
                    self.send_header("Cache-Control", "no-store")
                    if self.path == "/snapshot.jpg":
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(item[1])))
                        self.end_headers()
                        self.wfile.write(item[1])
                        return
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    while item:
                        seq, jpeg = item
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                         str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                        self.wfile.flush()
                        item = hub.wait(seq)
                except (OSError, TimeoutError):
                    pass
                finally:
                    slots.release()

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.hub.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class UartObserver:
    def __init__(self, port, baud, rate):
        import serial
        # A dedicated telemetry receiver is required. Never opens a default device.
        self.serial = serial.Serial(port=None, baudrate=baud, timeout=0,
                                    write_timeout=0.1, exclusive=True)
        self.serial.dtr = False
        self.serial.rts = False
        self.serial.port = port
        self.serial.open()
        self.period = 1 / rate
        self.last = -float("inf")

    def publish(self, obs):
        now = time.monotonic()
        if now - self.last < self.period:
            return
        packet = uart_packet(obs)
        if self.serial.out_waiting:
            return
        if self.serial.write(packet) != len(packet):
            raise RuntimeError("Partial UART telemetry write; stopping observer")
        self.last = now

    def close(self):
        self.serial.close()


class RosObserver:
    """Fixed observation topics; no Twist, control subscribers, TF, or legacy /vision topics."""
    def __init__(self, images=False):
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from std_msgs.msg import String
        from sensor_msgs.msg import CompressedImage

        self.rclpy = rclpy
        self.context = Context()
        self.node = None
        self.executor = None
        try:
            rclpy.init(args=[], context=self.context)
            self.node = Node("ugv_camera_observer", context=self.context, use_global_arguments=False)
            self.executor = SingleThreadedExecutor(context=self.context)
            self.executor.add_node(self.node)
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE)
            self.String = String
            self.CompressedImage = CompressedImage
            self.detections = self.node.create_publisher(String, "/camera_observer/detections", qos)
            self.status = self.node.create_publisher(String, "/camera_observer/status", qos)
            self.image = self.node.create_publisher(CompressedImage, "/camera_observer/image/compressed", qos) if images else None
        except BaseException:
            self.close()
            raise

    def publish(self, obs, jpeg=None):
        self.detections.publish(self.String(data=json_text(obs)))
        self.status.publish(self.String(data=json_text({"state": "running", "sequence": obs["sequence"],
                                                        "session_id": obs["session_id"]})))
        if self.image and jpeg:
            msg = self.CompressedImage()
            stamp = obs["capture_received_unix_ns"]
            msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
            # No unmeasured optical frame / extrinsic is asserted.
            msg.header.frame_id = ""
            msg.format = "jpeg"
            msg.data = jpeg
            self.image.publish(msg)
        self.executor.spin_once(timeout_sec=0)

    def terminal_status(self, state):
        self.status.publish(self.String(data=json_text({"state": state})))

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(timeout_sec=1)
        if self.node is not None:
            self.node.destroy_node()
        if self.context.ok():
            self.context.shutdown()

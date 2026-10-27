"""Pure configuration, geometry, and observation serialization."""

from dataclasses import dataclass
import json
import math
import zlib


@dataclass(frozen=True)
class CameraConfig:
    sensor_id: int = 0
    sensor_mode: int = 2
    capture_width: int = 1920
    capture_height: int = 1080
    capture_fps: int = 60
    width: int = 1280
    height: int = 720
    flip: int = 0

    def validate(self):
        if self.sensor_id < 0 or self.sensor_mode < 0:
            raise ValueError("Sensor ID/mode must be nonnegative")
        if not 0 <= self.flip <= 7:
            raise ValueError("Flip must be 0..7")
        for v in (self.capture_width, self.capture_height, self.width, self.height):
            if not 2 <= v <= 8192 or v % 2:
                raise ValueError("Dimensions must be even integers in 2..8192")
        if not 1 <= self.capture_fps <= 120:
            raise ValueError("Capture FPS must be 1..120; mode support is driver-specific")

    def pipeline(self):
        self.validate()
        return (
            f"nvarguscamerasrc sensor-id={self.sensor_id} sensor-mode={self.sensor_mode} ! "
            "video/x-raw(memory:NVMM),format=(string)NV12,"
            f"width=(int){self.capture_width},height=(int){self.capture_height},"
            f"framerate=(fraction){self.capture_fps}/1 ! "
            f"nvvidconv flip-method={self.flip} ! video/x-raw,"
            f"width=(int){self.width},height=(int){self.height},format=(string)BGRx ! "
            "videoconvert ! video/x-raw,format=(string)BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )


def finite_positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Expected a positive finite number")
    return value


def detection(class_id, label, confidence, xyxy, width, height, track_id=None):
    """Normalize boxes to actual received image dimensions; never estimate distance."""
    coords = [float(v) for v in xyxy]
    confidence = float(confidence)
    if len(coords) != 4 or not all(math.isfinite(v) for v in coords + [confidence]):
        raise ValueError("Non-finite/malformed detection")
    if not 0 <= confidence <= 1 or width <= 0 or height <= 0 or class_id < 0:
        raise ValueError("Invalid detection confidence/dimensions/class")
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Reversed or empty bounding box")
    x1, x2 = [min(float(width), max(0.0, v)) for v in (x1, x2)]
    y1, y2 = [min(float(height), max(0.0, v)) for v in (y1, y2)]
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Bounding box outside image")
    return {
        "class_id": int(class_id), "label": str(label), "confidence": confidence,
        "xyxy_px": [x1, y1, x2, y2], "track_id": track_id,
        "center_normalized": [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height)],
        "area_fraction": (x2 - x1) * (y2 - y1) / (width * height),
    }


def observation(session_id, sequence, stamp_ns, mono_ns, width, height, detections,
                inference_enabled=True):
    return {
        "schema": "ugv.camera.observation.v1", "session_id": session_id,
        "sequence": sequence, "capture_received_unix_ns": stamp_ns,
        "capture_received_monotonic_ns": mono_ns,
        "timestamp_basis": "host_after_capture_read_not_sensor_exposure",
        "image": {"width": width, "height": height, "coordinate_system": "image_x_right_y_down"},
        "inference_enabled": inference_enabled,
        "state": ("detections" if detections else "no_detections") if inference_enabled else "camera_only",
        "detections": detections,
    }


def json_text(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":"), ensure_ascii=True)


def uart_packet(obs):
    """New telemetry protocol, intentionally incompatible with legacy motion inputs."""
    compact = dict(obs)
    compact["detections"] = sorted(obs["detections"], key=lambda d: -d["confidence"])[:1]
    compact["total_detections"] = len(obs["detections"])
    payload = json_text(compact).encode("ascii")
    packet = b"UGVOBS1|" + payload + b"|" + f"{zlib.crc32(payload):08x}".encode() + b"\n"
    if len(packet) > 2048:
        raise ValueError("UART observation exceeds 2048 bytes")
    return packet


def parse_uart_packet(packet):
    if len(packet) > 2048 or not packet.endswith(b"\n"):
        raise ValueError("Invalid UART framing/length")
    prefix, body = packet[:-1].split(b"|", 1)
    payload, checksum = body.rsplit(b"|", 1)
    if prefix != b"UGVOBS1" or checksum != f"{zlib.crc32(payload):08x}".encode():
        raise ValueError("Invalid UART prefix/checksum")
    result = json.loads(payload)
    if result.get("schema") != "ugv.camera.observation.v1":
        raise ValueError("Unknown observation schema")
    return result

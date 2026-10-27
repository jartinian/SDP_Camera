# Observation interface contract

## Data meaning

The versioned JSON schema is `ugv.camera.observation.v1`. Each camera run has a random `session_id`; processed observations have a zero-based `sequence`. These count observations, not all sensor exposures. Queued old frames are deliberately dropped.

- `capture_received_unix_ns`: host wall-clock time immediately after OpenCV capture returns.
- `capture_received_monotonic_ns`: host monotonic time at that same point. Compare only within the same host boot/clock domain.
- `timestamp_basis`: explicitly says these are host receive times, not sensor exposure times.
- `image.width`/`height`: actual returned dimensions; coordinates are image x-right/y-down.
- `inference_enabled`: differentiates camera-only data from successful inference.
- `state`: `detections`, `no_detections`, or `camera_only`. Empty detections never mean a traversable scene or a safety clearance.
- `detections`: class ID, model label, confidence in [0,1], clipped `xyxy_px`, normalized center [0,1], area fraction [0,1], and nullable track ID.
- `processing_age_seconds`: age measured after inference; `published_unix_ns` is the host timestamp at observation construction, before optional encoding/output. Neither is a sensor latency measurement.

Non-finite confidences/boxes, reversed boxes, and boxes entirely outside the image cause an explicit error rather than permissive geometry. Distances, physical bearings, HFOV, stop decisions, steering directives and robot poses are intentionally absent. Bounding-box coordinates refer to the delivered, possibly flipped image.

A track ID is a per-session ByteTrack association, potentially null for unconfirmed tracks. It can change or be reassigned. It is not a stable person identity. There is no smoothed bearing carried between unrelated targets.

## Failure and freshness

The main loop rejects a frame older than `--max-frame-age` after inference and again before publication. This limit starts at host capture receipt and cannot detect time spent inside a camera driver before `read()` returned. Capture loss/timeout, malformed observations, inference failure, UART short writes, encoding failures and runtime recording errors end the session with a nonzero return and a fault summary. Recording initialization alone can be optional. EOF from file replay is a normal end.

Observation consumers must maintain their own receive-time watchdog and reject stale, malformed or incompatible schemas. An empty array is a valid inference result; silence means unknown/unavailable. Missing data must never be interpreted as “clear.” The default 2-second local observation-age limit is an adjustable tooling setting, not an approved vehicle safety requirement.

## ROS 2

Fixed publisher topics:

| Topic | Type | Purpose |
|---|---|---|
| `/camera_observer/detections` | `std_msgs/msg/String` | Full per-frame JSON observation |
| `/camera_observer/status` | `std_msgs/msg/String` | Running state/sequence/session and best-effort terminal status |
| `/camera_observer/image/compressed` | `sensor_msgs/msg/CompressedImage` | Optional annotated JPEG; host receive stamp |

All use best-effort, volatile, keep-last depth 1. No latched old detections. Default shutdown status is not guaranteed to reach subscribers, especially on crash, signal interruption or network loss: a subscriber watchdog is mandatory. A hung inference process cannot publish a fault until it returns; observation consumers and MJPEG stale checks must enforce their own deadlines.

The image `frame_id` is empty because no measured optical-frame/TF contract was provided. There is no `CameraInfo` or TF publication; a future calibrated adapter must add these explicitly. ROS image and JSON timestamps use system time; simulated ROS time is not supported. Do not use these fields as synchronized LiDAR exposure times without clock/latency calibration.

Node initialization uses an independent rclpy context, no global remapping arguments, and no subscriptions. No existing robot node is started. The current environment's ROS domain and RMW settings apply. The adapter is not an authentication boundary: another application could still choose to interpret observations as commands, which requires separate review.

## UART telemetry

UART is opt-in, requires an explicit dedicated observation-receiver port and `--uart-observer-confirmed`, defaults to 115200 baud and 2 Hz, and limits rate to 5 Hz. Port ownership uses pyserial's POSIX exclusive mode, which cannot guarantee exclusion against every unrelated process or identify downstream hardware. No serial port was opened during validation.

Each ASCII line is:

```text
UGVOBS1|<compact JSON>|<8 lowercase hexadecimal CRC32 digits>\n
```

CRC32 covers exactly the JSON payload bytes. It detects corruption; it does not authenticate messages. Maximum complete line length is 2048 bytes. The payload carries the same schema/session/sequence/timestamp/image metadata, but only the highest-confidence current detection plus `total_detections`. Empty detections are sent on the same rate-limited schedule. This is explicitly not a legacy numeric CSV packet and not a motor command.

If the serial output buffer is nonempty the new observation is skipped; no backlog grows in Python. Writes have a 100 ms timeout; partial writes fail the session. There is no retry, reconnect, ACK wait or RX command handling. Receivers should bound line buffers, discard bad CRC/schema/length, and time out lost senders. After a sender failure, discard an incomplete line and resume only at a new valid prefix. There is no final stop command because this is not an actuator protocol.

DTR/RTS are requested inactive before opening, but driver/hardware behavior can still cause control-line transitions. Do not attach this to an actuator serial port. For a single central Jetson, direct ROS observations are the preferred integration.

## Web stream

`/` serves HTML, `/video` multipart MJPEG, `/snapshot.jpg` a JPEG. One producer publishes the latest encoded image; client count does not change camera acquisition or model calls. Four image clients are permitted. Clients wait on a condition rather than spinning. Capture receive time governs image freshness. Stale multipart feeds close; stale/unavailable snapshot requests return 503. Browser rendering may retain the last displayed image after disconnection: consumers must detect connection loss and must not use a displayed image as proof of live sensing.

This is a small trusted-network development server with no authentication/TLS. Keep the loopback default and use SSH forwarding. It is not intended to resist arbitrary public-internet traffic.

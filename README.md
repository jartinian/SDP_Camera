# Jetson CSI camera / YOLO observer toolkit

For the current **Jetson Orin Nano + Arducam IMX477 CSI + NVIDIA CUDA PyTorch + YOLOv8 + ROS 2 Humble** setup. All ten uploaded scripts have been inspected and replaced by a shared implementation. The locally installed `live_yolo_auto.py` was inspected too. See [the script-by-script review](docs/SCRIPT_REVIEW.md), [migration findings](docs/MIGRATION.md), [interface contract](docs/INTERFACES.md), and [validation record](docs/VALIDATION.md).

The default is **headless CUDA detection**, with no recording, window, web server, UART, or ROS. There is no motor control, `Twist`, `/cmd_vel` publisher, legacy `/vision` publisher, LiDAR modification, TF publisher, service installation, or robot bring-up. This is a perception development toolkit, not a verified autonomous-driving or safety system.

## Start in this directory

Extract the ZIP into a new directory alongside your existing `~/YOLO_scripts`. Keep the whole toolkit directory together. Do not overwrite the migration repository or its ROS workspace. No installation or global configuration change is required if your current camera/YOLO dependencies already work.

```bash
cd ~/jetson_camera_toolkit
python3 camera_toolkit.py --help
python3 camera_toolkit.py --diagnose
python3 camera_toolkit.py --print-pipeline
```

Adjust the `cd` path to where you extracted it. Run the same Python interpreter/environment that successfully loaded NVIDIA PyTorch in your previous CSI test. Models are deliberately not bundled or downloaded automatically. Use your existing trusted weights:

```bash
python3 camera_toolkit.py --model ~/YOLO_scripts/yolov8n.pt --frames 100
```

`--device 0` is the default and fails if CUDA is unavailable. `--device cpu` is available explicitly for offline testing; it is not a silent fallback. A model warm-up occurs before opening the camera. Tracking also needs the `lap` dependency already installed. Runtime dependency auto-installation is disabled.

## Camera settings

Defaults use the sensor mode reported in your log: `sensor-id=0`, `sensor-mode=2`, **1920×1080 at 60 fps**, scaled to **1280×720 BGR**. The physical CAM1 connector does not imply Argus `sensor-id=1`. These settings reflect the supplied log, not a new hardware verification. They can be changed through `--sensor-id`, `--sensor-mode`, `--capture-width`, `--capture-height`, `--capture-fps`, `--width`, `--height`, and `--flip`. A change still must match the installed sensor driver's modes.

Only one process should own the CSI camera. If an existing camera server is using it, stop that camera owner deliberately before running this toolkit. No processes or Argus services are stopped automatically. Camera acquisition runs in a child process so a blocked native capture can be timed out and terminated without blocking observation shutdown indefinitely. Normal release is attempted first. A forced capture termination may require a later manual Argus recovery.

## Use cases

Run **one** toolkit process; combine flags to share one camera and one inference pass.

```bash
# Camera-only diagnostic: no model and no CUDA requirement
python3 camera_toolkit.py --no-inference --frames 30

# Desktop preview (requires a working display; q exits)
python3 camera_toolkit.py --model ~/YOLO_scripts/yolov8n.pt --show

# Record annotated video with MP4 -> AVI software fallback
python3 camera_toolkit.py --model ~/YOLO_scripts/yolov8n.pt --record auto --seconds 30

# Browser feed (loopback only by default)
python3 camera_toolkit.py --model ~/YOLO_scripts/yolov8n.pt --stream

# Real persistent ByteTrack IDs, optionally restricted to COCO stop signs
python3 camera_toolkit.py --model ~/YOLO_scripts/yolov8n.pt --track --classes 11 --record auto

# Shared tracking, preview stream, and recording
python3 camera_toolkit.py --model ~/YOLO_scripts/yolov8n.pt --track --stream --record auto

# Replay a video without camera hardware
python3 camera_toolkit.py --source-file /path/to/video.avi --no-inference --frames 30
```

Class 11 means stop sign only for the standard COCO YOLOv8 weights; use your model's own class mapping for custom weights. Tracking IDs are session-local associations and can switch after occlusion. They are not stable identities or navigation commands.

Open `http://127.0.0.1:8080/` on the Jetson desktop. The image routes are `/video` (MJPEG) and `/snapshot.jpg`; `/` is HTML, **not a video input URL**. For remote viewing, use an existing SSH connection with port forwarding:

```bash
ssh -N -L 8080:127.0.0.1:8080 sdp2026@YOUR_JETSON_HOST
```

Then open `http://127.0.0.1:8080/` on that computer. Explicit `--host 0.0.0.0` exposes an unauthenticated HTTP stream to reachable network peers; use it only on a controlled network. There is no HTTPS, RTSP, or audio server. Four simultaneous image clients are supported; they share encoded frames and do not run inference. Stale image streams close, and new stale snapshot requests return HTTP 503.

## Recording and outputs

Every run creates a uniquely named folder under `runs/camera/` (or `--output DIR`):

- `session.json`: arguments, session ID, OpenCV version, capture configuration.
- `observations.jsonl`: timestamped detections, including empty detections or camera-only state.
- `summary.json`: termination reason, error, processed/recorded counts.
- `video.mp4` or `video.avi`, when enabled, plus `video_timestamps.csv`.
- `recording_error.txt` if optional recording could not initialize.

`--record auto` tests a software `mp4v`/MP4 write and decode, then MJPEG/AVI. `--record avi` directly selects MJPEG. Neither uses NVIDIA NVENC. If initialization fails, inference continues with a clear warning; use `--record-required` to make it fatal. Errors during recording terminate the session; they are not silently ignored. A final first-frame decode catches gross corruption, but does not certify every frame. SIGINT/SIGTERM normally finalize the writer; power loss/SIGKILL can damage a video.

**Video playback uses `--record-fps` (default 10). One frame is written per processed observation.** Playback duration can therefore differ from wall time. Use the CSV and JSONL timestamps for measurements; no fabricated frames or unbounded catch-up loops are used. Replay is paced and drops queued older frames, just like live capture; it is not exhaustive offline processing.

Recording and logs can be large. The toolkit stops when available space falls below `--min-free-mb` (default 256); it does not delete or rotate files. Use `--seconds`/`--frames` for bounded experiments. Average FPS includes capture/output overhead, not only neural-network inference. Sensor exposure timestamps and calibrated sensor latency are unavailable through this OpenCV path.

## ROS 2 Humble: observation only

```bash
source /opt/ros/humble/setup.bash
python3 camera_toolkit.py --model ~/YOLO_scripts/yolov8n.pt --ros --ros-images
```

This publishes `std_msgs/msg/String` JSON on `/camera_observer/detections` and `/camera_observer/status`. Optional JPEG images use `sensor_msgs/msg/CompressedImage` on `/camera_observer/image/compressed`. QoS is best effort, volatile, depth 1; subscribers must use compatible QoS. It does not publish legacy vision topics or require a UART loopback on this now-central Jetson.

The current shell's ROS domain is inherited; nothing modifies it. These are development observation messages, not a drop-in `vision_msgs` detector or sensor-fusion node. Timestamps are host receive times; no camera calibration, mounting TF, distance, bearing, or camera-to-LiDAR association is invented. See [the interface contract](docs/INTERFACES.md), including stale-input handling.

## Optional UART telemetry

Prefer direct ROS observations on the central Jetson. A separate UART telemetry receiver can be used only with an explicitly selected, independently verified dedicated port:

```bash
python3 camera_toolkit.py --model ~/YOLO_scripts/yolov8n.pt \
  --uart-port /dev/YOUR_DEDICATED_OBSERVER_PORT --uart-observer-confirmed
```

The placeholder is deliberately not an assumed hardware path. Never point this at the motor controller, existing ESP bridge, LiDAR, or a port owned by another process. Opening a serial device can affect hardware control lines even without motor commands. This option requires a purpose-built receiver for `UGVOBS1` JSON telemetry. It is intentionally incompatible with the old comma-separated motion-associated formats. No ACK is required, no incoming bytes are interpreted, and no actuator command is generated. `ugv_camera.core.parse_uart_packet()` validates framing and CRC for an observer application. Physical UART behavior is not yet hardware verified.

## Environment preservation

Keep the working JetPack/NVIDIA PyTorch/torchvision installation and GStreamer-enabled OpenCV. **Do not install generic `torch`, `torchvision`, `opencv-python`, or `opencv-python-headless` over it.** Those packages can replace Jetson-specific builds or lose GStreamer/GUI support. The toolkit supplies no blanket pip requirements or installer for this reason.

The inspected environment reports Python 3.10.12, OpenCV 4.5.4 with GStreamer/FFmpeg, NVIDIA PyTorch `2.5.0a0+872d972e41.nv24.08`, and Ultralytics 8.3.248. The sandbox could not initialize CUDA; host diagnostics later confirmed CUDA available, and a bounded prerecorded CUDA inference test passed. The separate live CSI test failed in Argus; see the validation record. ROS needs the sourced Humble environment; UART needs existing pyserial. Track mode needs a compatible `lap` installation. Diagnose missing optional dependencies before changing this environment.

## Tests and troubleshooting

```bash
python3 -m compileall -q ugv_camera camera_toolkit.py tests
python3 -m unittest discover -s tests -v
```

Run these inside the extracted toolkit directory. Tests do not open camera, CUDA inference, motor UART, or the existing robot ROS graph. Synthetic recording uses installed OpenCV. The HTTP test opens only an ephemeral loopback port and is skipped if the sandbox disallows sockets.

- **CSI opens but MP4 fails:** use `--record auto`, `--record avi`, or leave recording off; see the review of `live_yolo_auto.py`.
- **No camera frames:** verify the camera is free, driver mode, Argus state, and `GStreamer: YES`. The tool exits on a bounded no-frame timeout, default 10 s.
- **CUDA unavailable:** use the known working Jetson Python environment, inspect `--diagnose`; do not solve it with a generic torch reinstall.
- **Stale frame:** the default 2 s age budget includes processing from host capture receipt. Benchmark without recording/streaming; an explicitly slower CPU replay may need `--max-frame-age 30`. This is an observation freshness limit, not an approved vehicle safety threshold.
- **Cannot open display:** omit `--show` and use the browser feed. Merely setting `DISPLAY` does not establish a working graphical session.
- **ROS missing:** source Humble in the same terminal; core camera functions do not require ROS.

The same-named legacy launchers in this bundle now call this toolkit. Their exact behavior changes are listed in the review. Original scripts are preserved byte-for-byte as `.py.txt` files under `legacy_reference/` to prevent accidental execution. Do not run those originals against connected actuators.

### Isolated tracking setup (optional)

The installed Ultralytics matching code requires `lap>=0.5.12`, and that dependency was absent during review. To add it without replacing the working system torch/OpenCV packages, create a separate environment that can see system packages:

```bash
python3 -m venv --system-site-packages .venv-tracking
.venv-tracking/bin/python -m pip install --no-deps 'lap>=0.5.12,<0.6'
.venv-tracking/bin/python camera_toolkit.py --diagnose
.venv-tracking/bin/python camera_toolkit.py --model ~/YOLO_scripts/yolov8n.pt --track --frames 100
```

This is an optional installation procedure, not something performed by the toolkit or this review. ARM64 wheel/build availability and tracking behavior still require target validation. Do not remove or upgrade the system NVIDIA libraries to satisfy this optional package.

### Optional ROS adapter test

Normal unit tests never initialize ROS. With Humble available, the additional adapter test can run on a separate localhost-only domain:

```bash
ROS_DOMAIN_ID=231 ROS_LOCALHOST_ONLY=1 UGV_TEST_ROS=1 python3 -m unittest discover -s tests -v
```

Choose a test environment where domain 231 is unused. The test checks observations and compressed-image delivery and absence of legacy command topics in that isolated graph. It does not validate integration with the existing robot graph.

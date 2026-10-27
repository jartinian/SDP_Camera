# Validation record — October 27, 2026

## Final outcome

**20 automated tests passed**, including a real localhost HTTP stream test and an isolated ROS 2 observation/image-delivery test. All **20 delivered Python files** parse and compile. Synthetic file replay, software recording/decoding, CPU YOLOv8 inference, and CUDA YOLOv8 inference passed their bounded smoke checks. These are functional checks, not accuracy, FPS, autonomy, or stopping-distance claims.

**Live CSI capture remains unresolved:** the bounded host camera-only test failed with Argus `Failed to create CaptureSession` and returned no frames. The toolkit reported a fault and exited in approximately 3.04 seconds. The earlier user-supplied log showed working CSI capture, but this run does not reproduce that success. No driver, service, serial device, LiDAR node, or motor node was changed/restarted to conceal the failure.

**Tracking dependency is absent:** installed Ultralytics matching code requires `lap>=0.5.12`; `lap` was not installed. Persistent tracking call semantics are unit-tested; actual ByteTrack runtime and real target-ID continuity were not run. An isolated optional dependency procedure is in the README. Physical UART and desktop GUI were not exercised.

## Environment

Host: Linux aarch64, Python 3.10.12, OpenCV 4.5.4 (GStreamer 1.19.90, FFmpeg, GTK3), NVIDIA PyTorch `2.5.0a0+872d972e41.nv24.08`, torchvision `0.20.0a0+afc54f7`, Ultralytics 8.3.248, NumPy 1.26.4, pyserial 3.5, ROS 2 Humble libraries. `host_diagnostics.log` confirms CUDA available outside the sandbox. `diagnostics.log` records the sandbox CUDA initialization failure; it does not establish a broken host install.

Existing migration repository: branch `fix/02-unitree-driver-and-tf`, commit `775a3577a63a8344646dfbd82dcf3dff30eaa554`, with pre-existing local changes. All 28 recorded migration documentation/interface source hashes stayed unchanged; final Git status matches the initially observed edited receiver and untracked bring-up files. No dependency installation or robot-stack code edit was performed.

## Test matrix

| Check | Result | Evidence / limitation |
|---|---|---|
| Original uploaded syntax audit | 9 parse; 1 fails | `original_scripts.json`; movement-stream line 231 |
| Delivered source syntax/compile | PASS, 20 files | `syntax_validation.log` |
| Core unit/recording tests in sandbox | PASS, HTTP skipped | `unittest.log`; initial suite of 19 tests |
| HTTP loopback host run | PASS, 19 tests | `unittest_loopback.log` |
| Final suite with isolated ROS test | PASS, 20 tests, zero skips | `unittest_host.log` |
| Initial ROS test | FAIL, fixed | `unittest_host_initial_failure.log`; default executor used an uninitialized global context; explicit per-context executors added and retested |
| Synthetic replay, MP4 write/read | PASS, 8 processed/recorded frames | `replay.log`, `replay_runs/`; software codec preflight and final decode; AVI roundtrip and forced MP4-fallback unit tests also pass |
| CPU YOLOv8 inference | PASS, 2 frames | `cpu_inference.log`, `cpu_runs/`; synthetic empty-detection input |
| Host CUDA diagnostic | PASS | `host_diagnostics.log` |
| CUDA YOLOv8 inference | PASS, 3 frames | `cuda_inference.log`, `cuda_runs/`; synthetic empty-detection input; NvMap allocation warnings occurred and are retained in the log |
| Live CSI capture | FAIL, clean fault exit | `csi_camera.log`, `csi_runs/`; Argus capture-session failure, zero frames |
| Real ByteTrack / target identity | NOT RUN | Missing lap; unit test verifies `persist=True`, explicit ByteTrack selection and one inference call |
| UART physical telemetry | NOT RUN | No serial port opened; CRC/framing/rate-limit/partial-write software tests pass |
| ROS integration with existing robot graph | NOT RUN | Adapter tested only in localhost-only domain 231, no actuator topics |
| Desktop live view | NOT RUN | No interactive display test |
| LiDAR/motor stack preservation | PASS for recorded files | `preservation_check.log`, source hashes and Git status; no autonomy test claimed |

The first unittest command was accidentally invoked from the parent directory and discovered zero tests. That result is not counted. The corrected commands below ran from the toolkit directory. The initial ROS failure was retained, repaired and rerun; it was not relabeled as a pass.

## Exact principal commands

Run from `jetson_camera_toolkit/`:

```bash
python3 -m compileall -q .
python3 -m unittest discover -s tests -v
ROS_DOMAIN_ID=231 ROS_LOCALHOST_ONLY=1 UGV_TEST_ROS=1 python3 -m unittest discover -s tests -v
python3 camera_toolkit.py --diagnose
python3 camera_toolkit.py --print-pipeline

python3 camera_toolkit.py --source-file evidence/synthetic_input.avi \
  --no-inference --frames 8 --record auto --record-required --output evidence/replay_runs

python3 camera_toolkit.py --source-file evidence/synthetic_input.avi \
  --model /home/sdp2026/YOLO_scripts/yolov8n.pt --device cpu --imgsz 160 \
  --frames 2 --max-frame-age 30 --output evidence/cpu_runs

timeout --signal=TERM --kill-after=5s 45s python3 camera_toolkit.py \
  --source-file evidence/synthetic_input.avi --model /home/sdp2026/YOLO_scripts/yolov8n.pt \
  --frames 3 --imgsz 320 --max-frame-age 10 --output evidence/cuda_runs

timeout --signal=TERM --kill-after=5s 35s python3 camera_toolkit.py \
  --no-inference --frames 5 --output evidence/csi_runs
```

The host-only checks were run outside the restricted execution sandbox; the ROS test set its domain/localhost options only for the test process. It used observation publishers and a test subscriber, not the existing robot graph. CUDA replay uses explicit default device 0 with no CPU fallback. Successful processing of synthetic blank/rectangle frames does not demonstrate detection accuracy on objects.

## Remaining camera acceptance

The next operator check is to identify the camera/Argus state and any existing owner on the actual host, then retry camera-only capture in the previously working terminal. Do not start multiple camera clients or change the IMX477 driver on the basis of this one error. Retain failed logs. Once capture works, rerun a bounded CUDA camera session and add optional outputs one at a time.

For tracking, install the isolated optional lap dependency and test real IDs through target entry, motion, occlusion and exit. For physical UART, verify a dedicated non-actuator observer connection and receive framing/rate/freshness. For future fusion, measure calibration, exposure/receipt latency and clock alignment. None of these pending checks proves safe motor behavior.

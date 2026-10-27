# Integration review and migration boundaries

## Sources and evidence

Inspected the existing `sensor-fusion-ugv-migration` repository at commit `775a3577a63a8344646dfbd82dcf3dff30eaa554`, branch `fix/02-unitree-driver-and-tf`. Its worktree already had a modified `vision_uart_receiver_node.py` plus untracked `start_bot.sh` and `start_bot.sh.save`. Those files and that repository were left untouched. Relevant files have hashes in `evidence/migration_source_hashes.json` so the inspected state can be identified.

Reviewed `AGENTS.md`, README, ADR-001, ADR-002, canonical architecture, roadmap, relevant robot-ready/verification material, Unitree interface contract, relevant `CODEBASE_DEEP_DIVE.md` sections, and the current vision UART receiver, camera follower, safety mux and ESP bridge implementations. Documentation was checked against current source at the vision/control boundary rather than treated as executable truth.

**Documentation says** — ADR-002/README target Raspberry Pi 4, Ubuntu 22.04, ROS 2 Humble, Python 3.10. That is historical compatibility evidence. The user now identifies Jetson Orin Nano as the central board and confirms removal of the Pi. This toolkit targets that newer stated arrangement without relabeling old Pi acceptance evidence as Jetson verification.

**Documentation says** — the roadmap records completed focused Phase 2 Unitree L2 checks and several pending later phases. The canonical architecture and older deep dive have some earlier pending/defect descriptions. These are dated evidence differences; this camera work does not resolve or overwrite them. It does not claim a complete current robot graph or a new LiDAR acceptance result.

## Current source findings that affect the camera

- **Verified from code** — the currently edited vision receiver defaults to `/dev/ttyTHS1`; the older deep dive describes `/dev/serial0`. Current source is the basis for this review. Neither path is adopted as a new toolkit default.
- **Verified from code** — receiver detection packets require exactly seven CSV fields with a numeric final angle. Separate two-character direction/stop packets and `NO_TARGET`/`NONE` are also recognized. The uploaded six-field, seven-field-string, and eight-field variants do not match that numeric contract.
- **Verified from code** — the receiver publishes `/vision/target_seen`, angle/confidence/box and other legacy topics. The camera follower consumes them and publishes a `Twist` candidate on `/cmd_vel_camera`; the mux uses the vision target-seen state and can select that candidate onto `/cmd_vel`. Merely publishing “vision data” onto those topics can affect motion.
- **Verified from code** — the follower uses a default 640×480 area denominator, assumes degree-like angles, negates the angle for steering, allows a lost-target grace period with forward creep, and smooths requested zero commands. Switching input images to 1280×720 without correcting the downstream geometry changes area ratios. The toolkit avoids that control path entirely.
- **Verified from code** — the ESP bridge subscribes to `/cmd_vel` and writes `CMD v ... w ...` serial lines. The toolkit imports no `Twist` and contains no publisher, subscriber, or command encoder for this interface.
- **Potential issue** — vision angle sign, lens calibration, camera-to-base/LiDAR extrinsics, receiver ownership, firmware motor behavior and safe thresholds remain unverified. Image x-right/y-down coordinates are explicitly a data convention, not a robot steering convention.

## New implemented observation path

```text
IMX477 -> Argus/GStreamer capture child -> one latest frame -> one YOLO inference
                                                     -> optional ByteTrack IDs
                                                     -> JSONL + timing metadata
                                                     -> optional view / recording / MJPEG
                                                     -> optional /camera_observer/* ROS topics
                                                     -> optional dedicated UGVOBS1 UART observer
```

One machine can run camera inference and ROS observations directly. There is no need to transmit camera detections from the Jetson back to itself through the old Pi-era UART path. New ROS topics are intentionally separate from `/vision/*` and every command candidate. There is no topic-remapping CLI for this adapter; its rclpy node ignores global ROS arguments. Users must not bridge these observations into actuation without a separately reviewed controller and safety validation.

The package does not modify/start the existing ESP bridge, LiDAR driver, projection, VFH, gap follow, fusion, mux, camera follower, or bring-up scripts. It does not install services, change ROS_DOMAIN_ID, alter networking, set clocks/power modes, change camera drivers, or reinstall NVIDIA packages. Original uploaded scripts are archived, not imported/executed.

## Deferred integration work

This is not completion of the repository's vision-autonomy phase. Before control integration, define and test the message schema used by a future controller, clock synchronization and freshness bounds, camera intrinsics/distortion, camera-to-LiDAR/base transforms, target selection/ID-loss policy, depth association, and coordinate conventions. A monocular box top/area is not a calibrated range or emergency-stop signal.

Any future motion path must preserve one final command owner, confirmed serial ownership, explicit enable, watchdogs, emergency-stop behavior and physical motor sign/speed validation. Those requirements come from the reviewed repository's architecture and safety instructions; no motion implementation or motor testing is included here.

## Migration steps

1. Retain the existing ROS workspace and working NVIDIA Python environment.
2. Extract this bundle into a separate directory; run syntax/unit tests and diagnostics.
3. When the CSI camera is free, run a bounded `--no-inference` capture.
4. Run a bounded CUDA YOLO session using the existing weights, with all optional outputs off.
5. Add recording, tracking, streaming and then isolated ROS observations one at a time. Record configuration and results.
6. Use the compatibility launchers only from the new directory. The imported `ugv_camera` directory must remain beside them. No original installation was overwritten by this work.

Primary technical references: [NVIDIA software encoding on Orin Nano](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/Multimedia/SoftwareEncodeInOrinNano.html), [Ultralytics persistent tracking](https://docs.ultralytics.com/modes/track). Local installed versions and runtime tests remain the acceptance basis; current online documentation may describe newer Ultralytics models/defaults. This implementation explicitly keeps YOLOv8 weights and ByteTrack.

#!/usr/bin/env python3
"""Continuously updating person/chair turn-away preview, revision 3.

Run alongside the camera with --classes 0 56. Requires the same Jetson monotonic
clock as the camera publisher. Publishes ONLY /camera_avoid/preview_cmd and
/camera_avoid/state. Do not remap to motors during the initial preview test.

This first stage NEVER requests forward motion. It does NOT drive past an object
or recover a route. Stops automatically retry after a one-second pause and fresh
camera data; each new turn requires three separate confirming frames. A turn
still lasts at most four seconds before pausing and reassessing. This automatic
retry behavior is for the motor-disconnected preview, not a validated drive mode.
Image areas/corridors below are uncalibrated trial settings, NOT metric clearance.
No detection does not prove free space. Test with motors disabled first.

V3 compares contiguous horizontal gaps between padded boxes rather than marking
an entire half-image occupied. Boxes ending above MIN_BOX_BOTTOM are excluded
from gap selection. This is an adjustable flat-floor/fixed-camera heuristic;
camera tilt, occlusion, or elevated obstacles can invalidate it. All large boxes
still trigger the close stop, regardless of their vertical position. Gaps are
image fractions, not measured traversable widths. No motor use is validated.
"""

import json
import math
import time
from dataclasses import dataclass


CONFIDENCE = 0.60
MIN_AREA = 0.015
TRIGGER_AREA = 0.03
CLOSE_AREA = 0.25
PATH_LEFT, PATH_RIGHT = 0.30, 0.70
BOX_MARGIN = 0.06
MIN_BOX_BOTTOM = 0.40  # Fraction down from image top; adjust for camera mounting.
MIN_GAP_WIDTH = 0.22   # Fraction of full image width, AFTER padding boxes.
GAP_PREFERENCE_MARGIN = 0.05
TURN_SPEED = 0.15
MAX_TURN_SECONDS = 4.0
RETRY_PAUSE_SECONDS = 1.0
MAX_FRAME_AGE = 0.50
MAX_RECEIVE_AGE = 0.50


@dataclass(frozen=True)
class Obstacle:
    class_id: int
    area: float
    left: float
    right: float
    bottom: float

    @property
    def x(self):
        return (self.left + self.right) / 2.0

    def overlaps(self, left, right):
        return self.left - BOX_MARGIN <= right and self.right + BOX_MARGIN >= left


def number(value):
    if isinstance(value, bool):
        raise ValueError("Boolean in numeric field")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Non-finite number")
    return value


def parse_obstacles(obs):
    # These fields are from ugv.camera.observation.v1 in camera_toolkit.
    width = number(obs["image"]["width"])
    height = number(obs["image"]["height"])
    if width <= 0 or height <= 0 or not isinstance(obs["detections"], list):
        raise ValueError("Invalid image/detections")
    obstacles = []
    for d in obs["detections"]:
        if not isinstance(d, dict) or type(d.get("class_id")) is not int:
            raise ValueError("Invalid detection")
        if d["class_id"] not in (0, 56):
            continue
        confidence = number(d["confidence"])
        area = number(d["area_fraction"])
        if not 0 <= confidence <= 1 or not 0 < area <= 1:
            raise ValueError("Invalid confidence/area")
        x1, y1, x2, y2 = [number(v) for v in d["xyxy_px"]]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("Invalid box")
        if confidence >= CONFIDENCE and area >= MIN_AREA:
            obstacles.append(Obstacle(d["class_id"], area, x1 / width,
                                      x2 / width, y2 / height))
    return obstacles


def widest_gap(boxes, start, end):
    """Largest uncovered interval in [start, end], after unioning padded boxes.

    It only means no selected detection covers those image columns. It is not
    evidence that undetected objects, walls, or obstacles beside the wheels are
    absent. Overlapping boxes must not be counted twice or split a real gap.
    """
    intervals = sorted(
        (max(start, b.left - BOX_MARGIN), min(end, b.right + BOX_MARGIN))
        for b in boxes
        if b.right + BOX_MARGIN > start and b.left - BOX_MARGIN < end
    )
    cursor = start
    best = (start, start)
    for left, right in intervals:
        if left - cursor > best[1] - best[0]:
            best = (cursor, left)
        cursor = max(cursor, right)
    if end - cursor > best[1] - best[0]:
        best = (cursor, end)
    return best


class AvoidanceController:
    """Pure decision logic: limited turns with automatic stopped reassessment."""

    def __init__(self):
        self.session = None
        self.sequence = -1
        self.stamp = None
        self.received = None
        self.last_processed = -1
        self.obstacles = []
        self.direction = 0  # -1 image-left; +1 image-right (tested follower mapping)
        self.started = None
        self.turn = 0.0
        self.confirm_count = 0
        self.pending_direction = 0
        self.clear_count = 0
        self.halt = None
        self.halted_at = None
        self.halted_frame = None
        self.state = "WAIT_CAMERA"
        self.scene_summary = "waiting for fresh frame"

    def latch(self, reason, now):
        # Remember the frame that caused a stop. Reusing that frame cannot
        # release the stop, even if the control timer runs several times.
        self.halt = reason
        self.halted_at = now
        self.halted_frame = (self.session, self.sequence)
        self.turn = 0.0

    def reset_trial(self):
        self.direction = 0
        self.started = None
        self.turn = 0.0
        self.confirm_count = 0
        self.pending_direction = 0
        self.clear_count = 0
        self.halt = None
        self.halted_at = None
        self.halted_frame = None

    def receive(self, data, now=None):
        now = time.monotonic() if now is None else now
        try:
            obs = json.loads(data)
            if not isinstance(obs, dict) or obs.get("schema") != "ugv.camera.observation.v1":
                raise ValueError("Invalid schema")
            if obs.get("inference_enabled") is not True:
                raise ValueError("Inference disabled")
            stamp_ns = obs["capture_received_monotonic_ns"]
            seq = obs["sequence"]
            session = obs["session_id"]
            if type(stamp_ns) is not int or type(seq) is not int or seq < 0:
                raise ValueError("Invalid timestamp/sequence")
            if not isinstance(session, str) or not session:
                raise ValueError("Invalid session")
            stamp = stamp_ns / 1e9
            if not 0 <= now - stamp < MAX_FRAME_AGE:
                raise ValueError("Stale frame")
            changed_session = self.session is not None and session != self.session
            if not changed_session and seq <= self.sequence:
                return  # Duplicates never refresh freshness or confirmation counts.
            if not changed_session and self.stamp is not None and stamp <= self.stamp:
                raise ValueError("Non-increasing capture timestamp")
            obstacles = parse_obstacles(obs)
            self.session, self.sequence = session, seq
            self.stamp, self.received = stamp, now
            self.obstacles = obstacles
            if changed_session:
                self.last_processed = -1
                self.latch("CAMERA_RESTART_STOP", now)
        except (ValueError, TypeError, KeyError, OverflowError, IndexError):
            self.received = None
            self.latch("INVALID_OR_STALE_DATA_STOP", now)

    def step(self, now=None):
        result = self.decide(now)
        self.state = result[2]
        return result

    def decide(self, now=None):
        now = time.monotonic() if now is None else now
        if self.halt:
            usable = (self.received is not None and self.stamp is not None
                      and 0 <= now - self.received < MAX_RECEIVE_AGE
                      and 0 <= now - self.stamp < MAX_FRAME_AGE)
            newer_frame = (self.session, self.sequence) != self.halted_frame
            if not usable or not newer_frame:
                return 0.0, 0.0, self.halt + "_WAIT_FRESH_CAMERA"
            if now - self.halted_at < RETRY_PAUSE_SECONDS:
                return 0.0, 0.0, self.halt + "_PAUSING"
            self.reset_trial()
            self.last_processed = self.sequence
            return 0.0, 0.0, "AUTO_RESET_RECHECKING"
        if self.received is None:
            return 0.0, 0.0, "WAIT_CAMERA"
        if (not 0 <= now - self.received < MAX_RECEIVE_AGE
                or not 0 <= now - self.stamp < MAX_FRAME_AGE):
            self.latch("CAMERA_TIMEOUT_STOP", now)
            return self.step(now)
        if self.started is not None and now - self.started >= MAX_TURN_SECONDS:
            self.latch("TURN_TIME_LIMIT_STOP", now)
            return self.step(now)

        fresh = self.sequence != self.last_processed
        if not fresh:
            return 0.0, self.turn, self.state
        self.last_processed = self.sequence
        all_boxes = self.obstacles
        boxes = [b for b in all_boxes if b.bottom >= MIN_BOX_BOTTOM]
        left_gap = widest_gap(boxes, 0.0, 0.5)
        right_gap = widest_gap(boxes, 0.5, 1.0)
        left_width = left_gap[1] - left_gap[0]
        right_width = right_gap[1] - right_gap[0]
        self.scene_summary = (
            f"selected_boxes={len(boxes)} | high_image_boxes={len(all_boxes)-len(boxes)}"
            f" | gap_left={left_width:.0%} | gap_right={right_width:.0%}"
        )

        if any(b.area >= CLOSE_AREA for b in all_boxes):
            self.latch("TOO_CLOSE_STOP", now)
            return self.step(now)

        central = [b for b in boxes if b.overlaps(PATH_LEFT, PATH_RIGHT)]
        # A side can contain an object AND an opening wide enough for this
        # preview. Require one contiguous gap, not the sum of narrow gaps.
        left_available = left_width >= MIN_GAP_WIDTH
        right_available = right_width >= MIN_GAP_WIDTH

        if self.direction == 0:
            triggers = [b for b in central if b.area >= TRIGGER_AREA]
            if not triggers:
                self.confirm_count = 0
                self.pending_direction = 0
                return 0.0, 0.0, "WAIT_NEAR_OBSTACLE_IN_PATH"
            if not left_available and not right_available:
                self.latch("NO_WIDE_VISUAL_GAP_STOP", now)
                return self.step(now)
            target = max(triggers, key=lambda b: b.area)
            # Original follower: +0.8*(x-0.5). Reversing its response means
            # positive for a left-of-center obstacle, negative for a right one.
            # A centered obstacle explicitly defaults to image-left (negative).
            direction = 1 if target.x < 0.45 else -1
            if not left_available:
                direction = 1
            elif not right_available:
                direction = -1
            elif right_width - left_width > GAP_PREFERENCE_MARGIN:
                direction = 1
            elif left_width - right_width > GAP_PREFERENCE_MARGIN:
                direction = -1
            self.confirm_count = self.confirm_count + 1 if direction == self.pending_direction else 1
            self.pending_direction = direction
            if self.confirm_count < 3:
                return 0.0, 0.0, "CONFIRM_OBSTACLE"
            self.direction = direction
            self.started = now

        # Keep the chosen direction for this trial; never flip left/right as
        # boxes jitter. A new obstruction stops the turn before reassessment.
        if (self.direction < 0 and not left_available) or (self.direction > 0 and not right_available):
            self.latch("CHOSEN_VISUAL_GAP_CLOSED_STOP", now)
            return self.step(now)
        if not boxes:
            self.latch("OBSTACLE_LOST_STOP" if not all_boxes
                       else "OBSTACLE_OUTSIDE_NEAR_ZONE_STOP", now)
            return self.step(now)
        if not central:
            self.clear_count += 1
            self.turn = 0.0
            if self.clear_count >= 3:
                self.latch("OBJECT_OUT_OF_CENTER_STOP", now)
                return self.step(now)
            return 0.0, 0.0, "CHECK_OBJECT_OUT_OF_CENTER"
        self.clear_count = 0
        self.turn = 0.65 * self.turn + 0.35 * (self.direction * TURN_SPEED)
        return 0.0, self.turn, "TURN_AWAY_" + ("LEFT" if self.direction < 0 else "RIGHT")


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import String
    from geometry_msgs.msg import Twist

    class CameraAvoidPreview(Node):
        def __init__(self):
            super().__init__("camera_avoid_preview")
            self.controller = AvoidanceController()
            self.cmd_pub = self.create_publisher(Twist, "/camera_avoid/preview_cmd", 10)
            self.state_pub = self.create_publisher(String, "/camera_avoid/state", 10)
            self.create_subscription(String, "/camera_observer/detections", self.on_detection, qos_profile_sensor_data)
            self.last_log = 0.0
            self.last_state = None
            self.create_timer(0.1, self.control_loop)
            self.get_logger().info("PREVIEW V3: compare visual gaps; filter high-image boxes; automatic retry. Forward always zero. Keep motor bridge disconnected.")

        def publish(self, forward, turn, state):
            cmd = Twist()
            cmd.linear.x = forward
            cmd.angular.z = turn
            self.cmd_pub.publish(cmd)
            self.state_pub.publish(String(data=state))
            now = time.monotonic()
            if state != self.last_state or now - self.last_log >= 1.0:
                self.get_logger().info(f"{state} | forward={forward:.3f} | turn={turn:.3f} | {self.controller.scene_summary}")
                self.last_state, self.last_log = state, now

        def on_detection(self, msg):
            self.controller.receive(msg.data)
            if self.controller.halt:
                self.publish(*self.controller.step())

        def control_loop(self):
            self.publish(*self.controller.step())

    rclpy.init()
    node = CameraAvoidPreview()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.cmd_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Smooth person-following preview for the Jetson camera toolkit.

Subscribes to camera observations and generates low-speed
person-following commands.

Publishes ONLY to isolated preview topics.
Does not connect directly to the motors.
"""

import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from geometry_msgs.msg import Twist


class PersonFollowPreview(Node):

    def __init__(self):
        super().__init__("person_follow_preview")

        # Camera observations
        self.create_subscription(
            String,
            "/camera_observer/detections",
            self.on_detection,
            qos_profile_sensor_data,
        )

        # Isolated output topics
        self.cmd_pub = self.create_publisher(
            Twist,
            "/camera_person_follow/preview_cmd",
            10,
        )

        self.state_pub = self.create_publisher(
            String,
            "/camera_person_follow/state",
            10,
        )

        # Camera state
        self.obs = None
        self.last_received = None
        self.last_session = None
        self.last_sequence = -1

        # Person tracking
        self.target_x = None
        self.filtered_x = None

        # Movement smoothing
        self.previous_turn = 0.0
        self.previous_forward = 0.0

        # Close-range stopping
        self.close_latched = False
        self.close_clear_count = 0
        self.last_close_sequence = None

        # Logging
        self.last_log = 0.0

        # Controller runs at 10 Hz
        self.create_timer(0.1, self.control_loop)

        self.get_logger().info(
            "Smooth person-following preview started. "
            "NO DIRECT MOTOR CONNECTION."
        )

    @staticmethod
    def clamp(value, low, high):
        return max(low, min(high, value))

    def reset_target(self):
        self.target_x = None
        self.filtered_x = None
        self.previous_turn = 0.0
        self.previous_forward = 0.0

    def on_detection(self, msg):
        try:
            obs = json.loads(msg.data)

            if obs.get("schema") != "ugv.camera.observation.v1":
                raise ValueError("Invalid camera schema")

            if not obs.get("inference_enabled", False):
                raise ValueError("Inference disabled")

            if not isinstance(obs.get("detections"), list):
                raise ValueError("Invalid detections")

            stamp = int(
                obs["capture_received_monotonic_ns"]
            )

            age = (
                time.monotonic_ns() - stamp
            ) / 1e9

            if not 0 <= age < 0.7:
                raise ValueError("Stale camera frame")

            session = obs["session_id"]
            sequence = int(obs["sequence"])

            if session != self.last_session:
                self.reset_target()
                self.last_session = session
                self.last_sequence = -1
                self.last_close_sequence = None
                self.close_clear_count = 0

            if sequence <= self.last_sequence:
                return

            self.last_sequence = sequence
            self.obs = obs
            self.last_received = time.monotonic()

        except (
            ValueError,
            TypeError,
            KeyError,
            OverflowError,
        ):
            self.obs = None
            self.last_received = None
            self.reset_target()

    def publish(self, forward, turn, state):
        cmd = Twist()

        cmd.linear.x = float(forward)
        cmd.angular.z = float(turn)

        self.cmd_pub.publish(cmd)
        self.state_pub.publish(
            String(data=state)
        )

        now = time.monotonic()

        if now - self.last_log >= 1.0:
            self.get_logger().info(
                f"{state} | "
                f"forward={forward:.3f} | "
                f"turn={turn:.3f}"
            )
            self.last_log = now

    def stop(self, state):
        # Safety stops must not be delayed by smoothing.
        self.previous_forward = 0.0
        self.previous_turn = 0.0

        self.publish(0.0, 0.0, state)

    def control_loop(self):
        now = time.monotonic()

        # -----------------------------------
        # CAMERA SAFETY CHECKS
        # -----------------------------------

        if self.obs is None or self.last_received is None:
            self.reset_target()
            self.stop("NO_CAMERA_DATA")
            return

        if now - self.last_received > 0.5:
            self.reset_target()
            self.stop("CAMERA_TIMEOUT")
            return

        # -----------------------------------
        # FIND VALID PEOPLE
        # -----------------------------------

        people = []

        for d in self.obs.get("detections", []):
            try:
                confidence = float(d["confidence"])
                area = float(d["area_fraction"])
                x = float(
                    d["center_normalized"][0]
                )

                if (
                    int(d["class_id"]) == 0
                    and all(
                        math.isfinite(v)
                        for v in (confidence, area, x)
                    )
                    and confidence >= 0.60
                    and 0.015 <= area <= 1.0
                    and 0.0 <= x <= 1.0
                ):
                    people.append((area, x))

            except (
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ):
                continue

        if not people:
            self.reset_target()
            self.stop("NO_PERSON")
            return

        people.sort(reverse=True)

        # -----------------------------------
        # TARGET SELECTION
        # -----------------------------------

        if self.target_x is None:

            # Do not choose between similarly sized people.
            if (
                len(people) > 1
                and people[1][0] >= 0.60 * people[0][0]
            ):
                self.stop("MULTIPLE_TARGETS_STOP")
                return

            area, x = people[0]

        else:

            nearby = sorted(
                people,
                key=lambda p: abs(
                    p[1] - self.target_x
                ),
            )

            area, x = nearby[0]

            nearest_error = abs(
                x - self.target_x
            )

            if nearest_error > 0.20:
                self.reset_target()
                self.stop("TARGET_CHANGED_STOP")
                return

            if len(nearby) > 1:
                second_error = abs(
                    nearby[1][1] - self.target_x
                )

                if second_error - nearest_error < 0.08:
                    self.reset_target()
                    self.stop("AMBIGUOUS_TARGET_STOP")
                    return

        self.target_x = x

        # -----------------------------------
        # SMOOTH PERSON POSITION
        # -----------------------------------

        if self.filtered_x is None:
            self.filtered_x = x
        else:
            self.filtered_x = (
                0.65 * self.filtered_x
                + 0.35 * x
            )

        # -----------------------------------
        # CLOSE-RANGE STOPPING
        # -----------------------------------
        # Bounding-box area is NOT a measured distance.

        if area >= 0.25:
            self.close_latched = True
            self.close_clear_count = 0

        if self.close_latched:

            # Count only distinct camera observations.
            sequence = (
                self.last_session,
                self.last_sequence,
            )

            if sequence != self.last_close_sequence:
                self.last_close_sequence = sequence

                if area < 0.18:
                    self.close_clear_count += 1
                else:
                    self.close_clear_count = 0

            # Require five fresh, sufficiently small detections
            # before releasing the close-range stop.
            if self.close_clear_count < 5:
                self.stop("TOO_CLOSE_STOP")
                return

            self.close_latched = False
            self.close_clear_count = 0

        # -----------------------------------
        # STEERING CALCULATION
        # -----------------------------------

        error = self.filtered_x - 0.5

        # Preserve the steering direction
        # confirmed during physical testing.

        if abs(error) <= 0.07:
            desired_turn = 0.0
        else:
            desired_turn = self.clamp(
                -0.8 * error,
                -0.20,
                0.20,
            )

        # Smooth angular corrections.
        turn = (
            0.65 * self.previous_turn
            + 0.35 * desired_turn
        )

        if abs(turn) < 0.008:
            turn = 0.0

        self.previous_turn = turn

        # -----------------------------------
        # FORWARD MOVEMENT CONTROL
        # -----------------------------------

        # Turn in place only for large errors.
        # Previously this threshold was 0.20.
        if abs(error) >= 0.30:
            self.previous_forward = 0.0
            self.publish(
                0.0,
                turn,
                "TURN_TO_PERSON",
            )
            return

        # Base speed based on apparent person size.
        if area >= 0.10:
            base_forward = 0.02
        else:
            base_forward = 0.04

        # Slow down while making a turn.
        # Keep some forward movement during moderate turns.
        turn_scale = self.clamp(
            1.0 - abs(error) / 0.45,
            0.50,
            1.0,
        )

        target_forward = (
            base_forward * turn_scale
        )

        # -----------------------------------
        # SMOOTH ACCELERATION
        # -----------------------------------

        # At 10 Hz, accelerate by no more
        # than 0.004 m/s per control cycle.
        # Deceleration remains immediate.

        forward = min(
            target_forward,
            self.previous_forward + 0.004,
        )

        self.previous_forward = forward

        self.publish(
            forward,
            turn,
            "FOLLOW_PERSON",
        )


def main():
    rclpy.init()
    node = PersonFollowPreview()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

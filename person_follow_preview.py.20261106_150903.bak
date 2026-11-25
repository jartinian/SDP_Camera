
#!/usr/bin/env python3
"""Person-following preview only. Never publishes motor commands."""

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

        self.create_subscription(
            String,
            "/camera_observer/detections",
            self.on_detection,
            qos_profile_sensor_data,
        )

        self.cmd_pub = self.create_publisher(
            Twist, "/camera_person_follow/preview_cmd", 10
        )
        self.state_pub = self.create_publisher(
            String, "/camera_person_follow/state", 10
        )

        self.obs = None
        self.last_received = None
        self.last_session = None
        self.last_sequence = -1

        self.target_x = None
        self.filtered_x = None
        self.previous_turn = 0.0

        self.close_latched = False
        self.close_clear_count = 0
        self.last_log = 0.0

        self.create_timer(0.1, self.control_loop)
        self.get_logger().info(
            "Improved person-follow preview. NO MOTOR CONNECTION."
        )

    @staticmethod
    def clamp(value, low, high):
        return max(low, min(high, value))

    def reset_target(self):
        self.target_x = None
        self.filtered_x = None
        self.previous_turn = 0.0

    def on_detection(self, msg):
        try:
            obs = json.loads(msg.data)

            if obs.get("schema") != "ugv.camera.observation.v1":
                raise ValueError("Invalid schema")
            if not obs.get("inference_enabled", False):
                raise ValueError("Inference disabled")

            stamp = int(obs["capture_received_monotonic_ns"])
            age = (time.monotonic_ns() - stamp) / 1e9

            if not 0 <= age < 0.7:
                raise ValueError("Stale frame")

            session = obs["session_id"]
            sequence = int(obs["sequence"])

            if session != self.last_session:
                self.reset_target()
                self.close_latched = False
                self.close_clear_count = 0
                self.last_session = session
                self.last_sequence = -1

            if sequence <= self.last_sequence:
                return

            self.last_sequence = sequence
            self.obs = obs
            self.last_received = time.monotonic()

        except (ValueError, TypeError, KeyError, OverflowError):
            self.obs = None
            self.last_received = None
            self.reset_target()

    def publish(self, forward, turn, state):
        cmd = Twist()
        cmd.linear.x = forward
        cmd.angular.z = turn

        self.cmd_pub.publish(cmd)
        self.state_pub.publish(String(data=state))

        now = time.monotonic()
        if now - self.last_log >= 1.0:
            self.get_logger().info(
                f"{state} | forward={forward:.3f} "
                f"| turn={turn:.3f}"
            )
            self.last_log = now

    def stop(self, state):
        self.previous_turn = 0.0
        self.publish(0.0, 0.0, state)

    def control_loop(self):
        now = time.monotonic()

        if self.obs is None or self.last_received is None:
            self.reset_target()
            self.stop("NO_CAMERA_DATA")
            return

        if now - self.last_received > 0.5:
            self.reset_target()
            self.stop("CAMERA_TIMEOUT")
            return

        people = []

        for d in self.obs.get("detections", []):
            try:
                confidence = float(d["confidence"])
                area = float(d["area_fraction"])
                x = float(d["center_normalized"][0])

                if (
                    int(d["class_id"]) == 0
                    and all(math.isfinite(v)
                            for v in (confidence, area, x))
                    and confidence >= 0.60
                    and 0.015 <= area <= 1.0
                    and 0.0 <= x <= 1.0
                ):
                    people.append((area, x))

            except (KeyError, IndexError, TypeError, ValueError):
                continue

        if not people:
            self.reset_target()
            self.stop("NO_PERSON")
            return

        people.sort(reverse=True)

        # Initial target: largest sufficiently distinct person.
        if self.target_x is None:
            if (
                len(people) > 1
                and people[1][0] >= 0.60 * people[0][0]
            ):
                self.stop("MULTIPLE_TARGETS_STOP")
                return

            area, x = people[0]

        else:
            # Maintain the previously selected person by position.
            nearby = sorted(
                people,
                key=lambda p: abs(p[1] - self.target_x)
            )

            area, x = nearby[0]
            nearest_error = abs(x - self.target_x)

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

        # Smooth image position.
        if self.filtered_x is None:
            self.filtered_x = x
        else:
            self.filtered_x = (
                0.65 * self.filtered_x + 0.35 * x
            )

        # Close-range stopping hysteresis.
        # Area is NOT a calibrated distance measurement.
        if area >= 0.25:
            self.close_latched = True
            self.close_clear_count = 0

        if self.close_latched:
            if area < 0.18:
                self.close_clear_count += 1
            else:
                self.close_clear_count = 0

            if self.close_clear_count < 3:
                self.stop("TOO_CLOSE_STOP")
                return

            self.close_latched = False
            self.close_clear_count = 0

        # Image center error.
        error = self.filtered_x - 0.5

        # Turning: positive left, negative right.
        if abs(error) <= 0.07:
            desired_turn = 0.0
        else:
            desired_turn = self.clamp(
                -0.8 * error, -0.20, 0.20
            )

        # Smooth steering.
        turn = (
            0.65 * self.previous_turn
            + 0.35 * desired_turn
        )
        if abs(turn) < 0.008:
            turn = 0.0
        self.previous_turn = turn

        if abs(error) >= 0.20:
            self.publish(0.0, turn, "TURN_TO_PERSON")
            return

        forward = 0.02 if area >= 0.10 else 0.04

        self.publish(forward, turn, "FOLLOW_PERSON")


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

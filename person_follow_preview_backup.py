
#!/usr/bin/env python3
"""Person-following preview. No motor commands or team code changes."""

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
        self.last_log = 0.0

        self.create_timer(0.1, self.control_loop)
        self.get_logger().info(
            "Person-follow preview running. Motors NOT connected."
        )

    def on_detection(self, msg):
        try:
            obs = json.loads(msg.data)
            if obs.get("schema") != "ugv.camera.observation.v1":
                raise ValueError("Unexpected observation schema")
            if not obs.get("inference_enabled", False):
                raise ValueError("Inference disabled")

            captured = int(obs["capture_received_monotonic_ns"])
            age = (time.monotonic_ns() - captured) / 1e9
            if not 0 <= age < 0.7:
                raise ValueError("Stale camera frame")

            self.obs = obs
            self.last_received = time.monotonic()
        except (ValueError, TypeError, KeyError, OverflowError):
            self.obs = None
            self.last_received = None

    def control_loop(self):
        cmd = Twist()
        state = "NO_PERSON"
        now = time.monotonic()

        if self.obs is None or self.last_received is None:
            state = "NO_CAMERA_DATA"

        elif now - self.last_received > 0.5:
            state = "CAMERA_TIMEOUT"

        else:
            people = []
            for d in self.obs.get("detections", []):
                try:
                    confidence = float(d["confidence"])
                    area = float(d["area_fraction"])
                    x = float(d["center_normalized"][0])
                    if (
                        d["class_id"] == 0
                        and all(math.isfinite(v) for v in (confidence, area, x))
                        and confidence >= 0.60
                        and 0.015 <= area <= 1.0
                        and 0 <= x <= 1
                    ):
                        people.append((area, x))
                except (KeyError, IndexError, TypeError, ValueError):
                    continue

            people.sort(reverse=True)

            if not people:
                state = "NO_PERSON"

            elif len(people) > 1 and people[1][0] >= 0.5 * people[0][0]:
                state = "MULTIPLE_TARGETS_STOP"

            else:
                area, x = people[0]
                error = x - 0.5

                if area >= 0.25:
                    state = "TOO_CLOSE_STOP"

                else:
                    if abs(error) > 0.07:
                        cmd.angular.z = max(
                            -0.20, min(0.20, -0.8 * error)
                        )

                    if abs(error) >= 0.20:
                        cmd.linear.x = 0.0
                        state = "TURN_TO_PERSON"
                    else:
                        cmd.linear.x = 0.02 if area >= 0.10 else 0.04
                        state = "FOLLOW_PERSON"

        self.cmd_pub.publish(cmd)
        self.state_pub.publish(String(data=state))

        if now - self.last_log >= 1.0:
            self.get_logger().info(
                f"{state} | forward={cmd.linear.x:.3f} "
                f"| turn={cmd.angular.z:.3f}"
            )
            self.last_log = now


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

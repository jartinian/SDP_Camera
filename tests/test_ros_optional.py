"""Opt-in real ROS loopback test; normal test runs never initialize ROS."""
import json
import os
import time
import unittest

from ugv_camera.core import observation


@unittest.skipUnless(os.environ.get("UGV_TEST_ROS") == "1", "Explicit ROS loopback test not enabled")
class RosLoopbackTest(unittest.TestCase):
    def test_observation_delivery(self):
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from std_msgs.msg import String
        from sensor_msgs.msg import CompressedImage
        from ugv_camera.outputs import RosObserver

        # The caller must deliberately isolate this test from the existing robot domain.
        self.assertEqual(os.environ.get("ROS_LOCALHOST_ONLY"), "1")
        self.assertEqual(os.environ.get("ROS_DOMAIN_ID"), "231")
        ctx = Context()
        rclpy.init(args=[], context=ctx)
        node = Node("camera_observer_test_subscriber", context=ctx, use_global_arguments=False)
        executor = SingleThreadedExecutor(context=ctx)
        executor.add_node(node)
        observer = None
        received, images = [], []
        try:
            node.create_subscription(String, "/camera_observer/detections",
                                     lambda m: received.append(json.loads(m.data)), qos_profile_sensor_data)
            node.create_subscription(CompressedImage, "/camera_observer/image/compressed",
                                     images.append, qos_profile_sensor_data)
            observer = RosObserver(images=True)
            obs = observation("test", 7, time.time_ns(), time.monotonic_ns(), 640, 480, [])
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not (received and images):
                observer.publish(obs, b"jpeg-test")
                executor.spin_once(timeout_sec=.05)
            self.assertTrue(received, "No observation arrived")
            self.assertEqual(received[0]["state"], "no_detections")
            self.assertEqual(received[0]["sequence"], 7)
            self.assertTrue(images, "No image arrived")
            self.assertEqual(bytes(images[0].data), b"jpeg-test")
            self.assertEqual(images[0].header.frame_id, "")
            topics = [name for name, types in node.get_topic_names_and_types()]
            self.assertNotIn("/cmd_vel", topics)
            self.assertNotIn("/vision/target_seen", topics)
        finally:
            if observer:
                observer.close()
            executor.shutdown(timeout_sec=1)
            node.destroy_node()
            ctx.shutdown()

#!/usr/bin/env python3
import math
import time
import serial

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from std_msgs.msg import String


def yaw_to_quat(yaw: float):
    half = yaw * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


class CmdVelToSerial(Node):
    def __init__(self):
        super().__init__('cmd_vel_to_serial')

        # Params
        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('rate_hz', 30.0)
        self.declare_parameter('rx_hz', 50.0)
        self.declare_parameter('debug_serial', False)
        self.declare_parameter('cmd_timeout_sec', 0.25)
        self.declare_parameter('max_linear_speed', 0.08)
        self.declare_parameter('max_angular_speed', 0.4)
        # Match unchanged v4 SENSOR_STALE_MS=800; diagnostic data only.
        self.declare_parameter('ultrasonic_timeout_sec', 0.8)
        self.cmd_timeout = float(self.get_parameter('cmd_timeout_sec').value)
        self.max_linear = float(self.get_parameter('max_linear_speed').value)
        self.max_angular = float(self.get_parameter('max_angular_speed').value)
        self.ultrasonic_timeout = float(self.get_parameter('ultrasonic_timeout_sec').value)
        self.serial_fault = False
        self._us_samples = {}
        self._us_published = None
        self._debug_enable_at = time.monotonic() + 1.0
        self._debug_enabled = False

        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('tof_front_topic', '/tof/front/summary')
        self.declare_parameter('ultra_raw_topic', '/esp/ultrasonic_raw')
        self.declare_parameter('motor_status_topic', '/esp/motor_status')

        port = self.get_parameter('port').get_parameter_value().string_value
        baud = self.get_parameter('baud').get_parameter_value().integer_value
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.rx_hz = float(self.get_parameter('rx_hz').value)
        self.debug_serial = bool(self.get_parameter('debug_serial').value)

        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        tof_front_topic = self.get_parameter('tof_front_topic').value
        ultra_raw_topic = self.get_parameter('ultra_raw_topic').value

        if any(not math.isfinite(x) or x <= 0 for x in
               (self.rate_hz, self.rx_hz, self.cmd_timeout, self.max_linear, self.max_angular,
                self.ultrasonic_timeout)):
            raise ValueError('Invalid bridge rates, limits, or timeout')
        self.ser = serial.Serial(port=None, baudrate=baud, timeout=0.0,
                                 write_timeout=0.05, exclusive=True)
        self.ser.dtr = False
        self.ser.rts = False
        self.ser.port = port
        self.ser.open()
        self.get_logger().info(f"Serial open: {port} @ {baud}")

        # cmd_vel in
        self.last_twist = Twist()
        self.last_rx_time = time.monotonic()
        self.sub = self.create_subscription(Twist, cmd_vel_topic, self.on_cmd_vel, 10)

        # odom out + TF
        self.odom_pub = self.create_publisher(Odometry, odom_topic, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # sensor outputs
        self.tof_front_pub = self.create_publisher(String, tof_front_topic, 10)
        self.ultra_raw_pub = self.create_publisher(String, ultra_raw_topic, 10)
        self.motor_status_pub = self.create_publisher(
            String, self.get_parameter('motor_status_topic').value, 10)

        # Serial RX line buffer
        self._rx_buf = ""

        # Timers
        self.cmd_timer = self.create_timer(1.0 / self.rate_hz, self.on_cmd_timer)
        self.rx_timer = self.create_timer(1.0 / self.rx_hz, self.on_rx_timer)

        self.get_logger().info(
            f"Bridge running: {cmd_vel_topic} -> serial, serial -> {odom_topic}, "
            f"{tof_front_topic}, {ultra_raw_topic}"
        )

    def on_cmd_vel(self, msg: Twist):
        values = (msg.linear.x, msg.linear.y, msg.linear.z,
                  msg.angular.x, msg.angular.y, msg.angular.z)
        if not all(math.isfinite(value) for value in values):
            self.last_twist = Twist()
            self.get_logger().error('Nonfinite cmd_vel rejected; requesting zero')
        else:
            self.last_twist = msg
        self.last_rx_time = time.monotonic()

    def send_zero(self):
        # Best effort only: a disconnected link cannot deliver a stop.
        try:
            self.ser.write(b'CMD v 0.000 w 0.000\n')
        except Exception:
            pass

    def on_cmd_timer(self):
        if self.serial_fault or time.monotonic() - self.last_rx_time > self.cmd_timeout:
            v, w = 0.0, 0.0
        else:
            v, w = float(self.last_twist.linear.x), float(self.last_twist.angular.z)
            if not math.isfinite(v) or not math.isfinite(w):
                v, w = 0.0, 0.0
            # No reverse in the initial LiDAR-only profile. Preserve curvature.
            v = max(0.0, v)
            factor = min(1.0, self.max_linear / max(abs(v), 1e-12),
                         self.max_angular / max(abs(w), 1e-12))
            v, w = v * factor, w * factor
        payload = f'CMD v {v:.3f} w {w:.3f}\n'.encode('ascii')
        try:
            # Re-enable v4 diagnostics after a possible USB-open board reset.
            # This command changes logging only; velocity comes solely from mux.
            if self.debug_serial and not self._debug_enabled and time.monotonic() >= self._debug_enable_at:
                debug = b'DEBUG 1\n'
                if self.ser.write(debug) != len(debug):
                    raise IOError('partial debug command write')
                self._debug_enabled = True
            if self.ser.write(payload) != len(payload):
                raise IOError('partial serial write')
            now = time.monotonic()
            if now - getattr(self, '_last_camera_tx_log', 0.0) >= 1.0:
                self._last_camera_tx_log = now
                self.get_logger().info(
                    f'CAMERA_TX v={v:.3f} w={w:.3f} '
                    f'age={now - self.last_rx_time:.3f}s '
                    f'fault={self.serial_fault}')
        except Exception as e:
            self.serial_fault = True
            self.last_twist = Twist()
            self.get_logger().error(f'Serial fault latched; restart bridge after repair: {e}')
            self.send_zero()

    def on_rx_timer(self):
        self.publish_v4_ultrasonic()
        try:
            data = self.ser.read(4096)
            if not data:
                return
            self._rx_buf += data.decode('ascii', errors='ignore')
        except Exception as e:
            self.serial_fault = True
            self.last_twist = Twist()
            self.send_zero()
            self.get_logger().error(f"Serial read fault latched: {e}")
            return

        if len(self._rx_buf) > 16384:
            self._rx_buf = ''
            self.get_logger().warn('Oversized serial receive buffer discarded')
            return

        while '\n' in self._rx_buf:
            line, self._rx_buf = self._rx_buf.split('\n', 1)
            line = line.strip()
            if not line:
                continue

            if self.debug_serial:
                self.get_logger().info(f"SERIAL: {line}")

            if line.startswith("ODOM"):
                self.handle_odom_line(line)
            elif line.startswith("TOF|"):
                self.handle_tof_line(line)
            elif line.startswith("ULTRA|"):
                self.handle_ultra_line(line)
            elif line.startswith("US|"):
                self.handle_v4_us_line(line)
            elif line.startswith("DBG|T:"):
                msg = String()
                msg.data = line
                self.motor_status_pub.publish(msg)
            # Default v4 emits no ODOM. Never synthesize it from commands.

    def handle_v4_us_line(self, line):
        """Convert unchanged v4's per-sensor US records to the ROS schema."""
        try:
            pairs = [part.split(':', 1) for part in line.split('|')[1:]]
            fields = dict(pairs)
            if len(fields) != len(pairs):
                raise ValueError('duplicate field')
            name = fields['ID']
            stamp = int(fields['T'])
            valid = int(fields['VALID'])
            cm = float(fields['CM'])
            if (name not in ('FRONT', 'LEFT', 'RIGHT', 'BACK') or
                    not 0 <= stamp <= 0xffffffff or valid not in (0, 1) or
                    not math.isfinite(cm) or not 2.0 <= cm <= 400.0):
                raise ValueError('invalid ultrasonic fields')
        except (KeyError, ValueError):
            self.get_logger().warn(f'Bad v4 US line: {line}')
            return
        previous = self._us_samples.get(name)
        # Repeated telemetry for the same MCU sample must not renew freshness.
        received = (previous[3] if previous and previous[0] == stamp
                    else time.monotonic())
        self._us_samples[name] = (stamp, cm, valid, received)
        self.publish_v4_ultrasonic()

    def publish_v4_ultrasonic(self):
        if not self._us_samples:
            return
        now = time.monotonic()
        distances, validity = {}, {}
        for name in ('FRONT', 'LEFT', 'RIGHT', 'BACK'):
            sample = self._us_samples.get(name)
            valid = bool(sample and sample[2] and now - sample[3] <= self.ultrasonic_timeout)
            distances[name] = sample[1] if valid else 400.0
            validity[name] = int(valid)
        # T is the most recently received MCU sample timestamp, including wrap.
        stamp = max(self._us_samples.values(), key=lambda sample: sample[3])[0]
        data = (f'T:{stamp}|' + '|'.join(f'{k}:{v:.1f}' for k, v in distances.items()) +
                '|' + '|'.join(f'VALID_{k}:{v}' for k, v in validity.items()))
        if data != self._us_published:
            msg = String()
            msg.data = data
            self.ultra_raw_pub.publish(msg)
            self._us_published = data

    def handle_odom_line(self, line: str):
        try:
            parts = line.split()

            def get_val(label: str) -> float:
                i = parts.index(label)
                return float(parts[i + 1])

            x = get_val("x")
            y = get_val("y")
            th = get_val("th")
            v = get_val("v")
            w = get_val("w")
            if not all(math.isfinite(value) for value in (x, y, th, v, w)):
                raise ValueError('Nonfinite odometry')
        except Exception:
            self.get_logger().warn(f"Bad ODOM line: {line}")
            return

        now = self.get_clock().now().to_msg()
        qx, qy, qz, qw = yaw_to_quat(th)

        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.twist.twist.linear.x = v
        odom.twist.twist.linear.y = 0.0
        odom.twist.twist.angular.z = w

        self.odom_pub.publish(odom)

        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(t)

    def handle_tof_line(self, line: str):
        try:
            parts = line.split('|')
            if parts[0] != "TOF":
                self.get_logger().warn(f"Bad TOF prefix: {line}")
                return

            fields = {}
            for part in parts[1:]:
                if ':' not in part:
                    continue
                key, value = part.split(':', 1)
                fields[key] = value

            sensor_id = fields.get("ID", "")
            if sensor_id != "FRONT":
                return

            timestamp_ms = int(fields.get("T", "0"))
            valid = int(fields.get("VALID", "0"))
            center = float(fields.get("CENTER", "0.0"))
            near_min = float(fields.get("NEAR_MIN", "0.0"))
            near_avg = float(fields.get("NEAR_AVG", "0.0"))
            far_avg = float(fields.get("FAR_AVG", "0.0"))
        except Exception:
            self.get_logger().warn(f"Bad TOF line: {line}")
            return

        msg = String()
        msg.data = (
            f"ID:{sensor_id}|T:{timestamp_ms}|VALID:{valid}|"
            f"CENTER:{center:.1f}|NEAR_MIN:{near_min:.1f}|"
            f"NEAR_AVG:{near_avg:.1f}|FAR_AVG:{far_avg:.1f}"
        )
        self.tof_front_pub.publish(msg)

    def handle_ultra_line(self, line: str):
        """
        Expected:
        ULTRA|T:12346|FRONT:22.6|LEFT:41.8|RIGHT:39.5|BACK:55.2|VALID_FRONT:1|VALID_LEFT:1|VALID_RIGHT:1|VALID_BACK:1
        """
        try:
            parts = line.split('|')
            if parts[0] != "ULTRA":
                self.get_logger().warn(f"Bad ULTRA prefix: {line}")
                return

            fields = {}
            for part in parts[1:]:
                if ':' not in part:
                    continue
                key, value = part.split(':', 1)
                fields[key] = value

            timestamp_ms = int(fields.get("T", "0"))
            front = float(fields.get("FRONT", "400.0"))
            left = float(fields.get("LEFT", "400.0"))
            right = float(fields.get("RIGHT", "400.0"))
            back = float(fields.get("BACK", "400.0"))

            valid_front = int(fields.get("VALID_FRONT", "0"))
            valid_left = int(fields.get("VALID_LEFT", "0"))
            valid_right = int(fields.get("VALID_RIGHT", "0"))
            valid_back = int(fields.get("VALID_BACK", "0"))
        except Exception:
            self.get_logger().warn(f"Bad ULTRA line: {line}")
            return

        msg = String()
        msg.data = (
            f"T:{timestamp_ms}|FRONT:{front:.1f}|LEFT:{left:.1f}|RIGHT:{right:.1f}|BACK:{back:.1f}|"
            f"VALID_FRONT:{valid_front}|VALID_LEFT:{valid_left}|VALID_RIGHT:{valid_right}|VALID_BACK:{valid_back}"
        )
        self.ultra_raw_pub.publish(msg)


def main():
    rclpy.init()
    node = CmdVelToSerial()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.send_zero()
        try:
            node.ser.close()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

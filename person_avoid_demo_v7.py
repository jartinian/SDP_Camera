#!/usr/bin/env python3
"""Camera-only continuous person avoidance demonstration, V7.

Default prints decisions only. --publish-preview sends identical decisions as
Twist on /camera_person_avoid/preview_cmd to the existing camera motor bridge.
No ultrasonic, LiDAR, odometry or direct serial access is used.

Fresh camera -> drive straight -> stop for a person -> confirm -> stationary
turn (at most 6 seconds) -> stopped visual recheck -> KEEP DRIVING on the new
heading. A new person detection in the center path starts another stop/turn
cycle. There is no elapsed-time stop for straight driving. The operator ends
the demonstration with motor power off / Ctrl+C; do not leave it unattended.

The recheck requires the encountered person to remain visible outside the padded
center corridor, on the opposite side from the turn, for 0.5 seconds and three
fresh frames. Loss before that point stops it. After departure begins, losing
that detection does not stop straight driving; fresh camera data is still
required and any detected person in the center corridor triggers a new stop.
Camera faults and unsuccessful turns latch a stop until restart. Blocked gaps
are stationary wait states that can recover if fresh frames show a valid gap.
Brief YOLO detection dropouts stop motion and allow reacquisition; longer loss
still latches a stop, since missing detections do not establish clearance.

This is an uncalibrated supervised demo, NOT measured clearance or a complete
route planner. Missing detections do not prove free space. It can miss people,
walls, stairs and other obstacles, including beside the wheels. Use only the
cleared demonstration area with a separate operator at the motor cutoff. It
continues on its NEW heading; it does not return to its original route.

One confidence threshold (0.35), no box-area veto. Physical clearance is checked
by the operator. Steering signs are specific to this user's tested robot.
A bridge watchdog is required; a broken serial link needs an MCU watchdog too.
Uses ugv.camera.observation.v1 and the SAME Jetson monotonic clock.
Run --self-test for simulated checks without ROS.
"""

import argparse
import json
import math
import time
from dataclasses import dataclass


FORWARD = 0.02
TURN = 0.15
FRAME_TIMEOUT = 0.5
MAX_TURN_SECONDS = 6.0
CLEAR_DWELL = 0.5
DETECTION_GRACE = 0.7  # stop immediately, permit short detection dropout
GAP_RECOVERY_FRAMES = 3  # fresh detections needed to replan a blocked gap
CONFIRM_TIMEOUT = 2.0
STOP_DWELL = 0.5
PATH_LEFT, PATH_RIGHT = 0.30, 0.70
PADDING = 0.06
MIN_GAP = 0.22
CAUTION_CONFIDENCE = 0.35


@dataclass(frozen=True)
class Person:
    left: float
    top: float
    right: float
    bottom: float
    confidence: float
    area: float

    @property
    def x(self):
        return (self.left + self.right) / 2

    @property
    def in_path(self):
        return self.left - PADDING <= PATH_RIGHT and self.right + PADDING >= PATH_LEFT


def finite(value):
    if isinstance(value, bool):
        raise ValueError('Boolean numeric field')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Nonfinite numeric field')
    return value


def parse_people(obs):
    width = finite(obs['image']['width'])
    height = finite(obs['image']['height'])
    if width <= 0 or height <= 0 or not isinstance(obs['detections'], list):
        raise ValueError('Invalid image or detections')
    people = []
    for d in obs['detections']:
        if not isinstance(d, dict) or type(d.get('class_id')) is not int:
            raise ValueError('Invalid detection')
        if d['class_id'] != 0:
            continue
        confidence = finite(d['confidence'])
        reported_area = finite(d['area_fraction'])
        x1, y1, x2, y2 = map(finite, d['xyxy_px'])
        if not (0 <= confidence <= 1 and 0 < reported_area <= 1
                and 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError('Invalid person box')
        area = (x2 - x1) * (y2 - y1) / (width * height)
        if confidence >= CAUTION_CONFIDENCE:
            people.append(Person(x1 / width, y1 / height, x2 / width,
                                 y2 / height, confidence, max(area, reported_area)))
    return people


def gap_width(people, start, end):
    intervals = sorted((max(start, p.left - PADDING), min(end, p.right + PADDING))
                       for p in people if p.right + PADDING > start
                       and p.left - PADDING < end)
    cursor, best = start, 0.0
    for left, right in intervals:
        best = max(best, left - cursor)
        cursor = max(cursor, right)
    return max(best, end - cursor)


def box_continuity(old, new):
    # Loose geometric continuity accommodates split/merged boxes. It is not
    # identity recognition and cannot disambiguate crossing/occluded people.
    overlap_x = max(0.0, min(old.right, new.right) - max(old.left, new.left))
    overlap_y = max(0.0, min(old.bottom, new.bottom) - max(old.top, new.top))
    smaller = min((old.right-old.left)*(old.bottom-old.top),
                  (new.right-new.left)*(new.bottom-new.top))
    return overlap_x * overlap_y / smaller >= 0.10 and abs(old.x-new.x) <= 0.20


class Controller:
    def __init__(self):
        self.session = None
        self.sequence = -1
        self.stamp = None
        self.processed = -1
        self.people = []
        self.phase = 'STARTUP'
        self.command = (0.0, 0.0, 'WAIT_CAMERA')
        self.halt_reason = None
        self.start_count = 0
        self.cruise_started = None
        self.encountered = None
        self.turn_started = None
        self.exit_started = None
        self.clear_started = None
        self.clear_count = 0
        self.tracked_boxes = []
        self.direction = 0
        self.confirm_count = 0
        self.turn = 0.0
        self.missing_since = None
        self.gap_open_count = 0
        self.summary = 'waiting for camera'

    def output(self, v, w, state):
        self.command = (v, w, state)
        return self.command

    def halt(self, reason):
        if self.halt_reason is None:
            self.halt_reason = reason
        self.phase = 'HALTED'
        return self.output(0.0, 0.0, self.halt_reason + '_RESTART_TO_RETRY')

    def receive(self, data, now):
        if self.halt_reason:
            return
        try:
            obs = json.loads(data)
            if not isinstance(obs, dict) or obs.get('schema') != 'ugv.camera.observation.v1':
                raise ValueError('Invalid schema')
            if obs.get('inference_enabled') is not True:
                raise ValueError('Inference disabled')
            stamp = obs['capture_received_monotonic_ns']
            seq = obs['sequence']
            session = obs['session_id']
            if (type(stamp) is not int or type(seq) is not int or seq < 0
                    or not isinstance(session, str) or not session):
                raise ValueError('Invalid frame metadata')
            stamp /= 1e9
            if not 0 <= now - stamp < FRAME_TIMEOUT:
                raise ValueError('Stale frame')
            if self.session is not None and session != self.session:
                self.halt('CAMERA_RESTART_STOP')
                return
            if seq <= self.sequence:
                return
            if self.stamp is not None and stamp <= self.stamp:
                raise ValueError('Nonincreasing capture timestamp')
            people = parse_people(obs)
            self.session, self.sequence, self.stamp = session, seq, stamp
            self.people = people
        except (ValueError, TypeError, KeyError, IndexError, OverflowError):
            self.halt('INVALID_CAMERA_DATA_STOP')

    def step(self, now):
        if self.halt_reason:
            return self.halt(self.halt_reason)
        if self.stamp is None:
            return self.output(0.0, 0.0, 'WAIT_CAMERA')
        if not 0 <= now - self.stamp < FRAME_TIMEOUT:
            return self.halt('CAMERA_TIMEOUT_STOP')
        if (self.phase in ('TURN', 'CHECK_EXIT') and self.turn_started is not None
                and now - self.turn_started >= MAX_TURN_SECONDS):
            return self.halt('TURN_TIME_LIMIT_STOP')
        if self.phase == 'CONFIRM' and now - self.encountered >= CONFIRM_TIMEOUT:
            return self.halt('UNCONFIRMED_PERSON_STOP')
        if self.sequence == self.processed:
            return self.command
        self.processed = self.sequence
        people = self.people
        left = gap_width(people, 0, 0.5)
        right = gap_width(people, 0.5, 1)
        central = [p for p in people if p.in_path]
        self.summary = (f'person_boxes={len(people)} | in_path={len(central)} | '
                        f'gap_left={left:.0%} | gap_right={right:.0%}')

        if self.phase in ('STARTUP', 'STRAIGHT'):
            if not central:
                if self.phase == 'STARTUP':
                    self.start_count += 1
                    if self.start_count < 5:
                        return self.output(0.0, 0.0, 'CHECK_FRESH_CAMERA')
                    self.phase = 'STRAIGHT'
                    self.cruise_started = now
                return self.output(FORWARD, 0.0, 'DRIVE_STRAIGHT')
            return self.start_avoidance(central, now)

        # Continue on the new heading after the visual recheck. A new obstacle
        # always stops forward motion immediately and starts a fresh turn cycle.
        # The absence of a box is not measured free space; supervision is needed.
        if self.phase == 'EXIT':
            if central:
                return self.start_avoidance(central, now)
            return self.output(FORWARD, 0.0, 'CONTINUE_STRAIGHT')

        # A blocked visual gap is not an emergency command to proceed; stay
        # stationary and reconsider on each new, fresh frame. This state may
        # only re-enter CONFIRM after a stable visible obstacle and viable gap.
        if self.phase == 'WAIT_GAP':
            if not central or (left < MIN_GAP and right < MIN_GAP):
                self.gap_open_count = 0
                return self.output(0.0, 0.0, 'WAIT_FOR_VISUAL_GAP')
            self.gap_open_count += 1
            if self.gap_open_count < GAP_RECOVERY_FRAMES:
                return self.output(0.0, 0.0, 'WAIT_GAP_RECHECK')
            self.phase = 'CONFIRM'
            self.encountered = now
            self.confirm_count = 0
            self.direction = 0
            self.turn = 0.0
            self.turn_started = None
            self.tracked_boxes = list(central)
            self.missing_since = None
            return self.output(0.0, 0.0, 'GAP_RECOVERED_CONFIRM_STOP')

        # Many-to-many continuity: a person may split into overlapping leg
        # boxes. Do not require an exact bounding box, but never treat an
        # unrelated person as proof that the encountered group moved aside.
        if not people:
            if self.missing_since is None:
                self.missing_since = now
            # Never keep turning or moving during a detection dropout.
            self.turn = 0.0
            self.clear_started = None
            self.clear_count = 0
            if self.phase == 'CHECK_EXIT':
                # A missing frame invalidates the visual-clear dwell. Rebuild
                # that evidence from fresh images after reacquisition.
                self.phase = 'TURN'
            if now - self.missing_since < DETECTION_GRACE:
                return self.output(0.0, 0.0, 'WAIT_PERSON_REDETECTION')
            return self.halt('PERSON_LOST_STOP')
        self.missing_since = None
        tracked = [p for p in people if any(box_continuity(old, p)
                                           for old in self.tracked_boxes)]
        if not tracked or any(not any(box_continuity(old, p) for p in people)
                              for old in self.tracked_boxes):
            return self.halt('ENCOUNTERED_PERSON_LOST_STOP')
        self.tracked_boxes = tracked

        if not central:
            if self.phase == 'CONFIRM':
                return self.halt('PERSON_MOVED_BEFORE_TURN_STOP')
            if not self.person_on_expected_side(tracked):
                return self.halt('PERSON_SIDE_CHANGED_STOP')
            if self.phase != 'CHECK_EXIT':
                self.phase = 'CHECK_EXIT'
                self.clear_started = now
                self.clear_count = 0
            self.clear_count += 1
            self.turn = 0.0
            if self.clear_count >= 3 and now - self.clear_started >= CLEAR_DWELL:
                self.phase = 'EXIT'
                self.exit_started = now
                return self.output(FORWARD, 0.0, 'CONTINUE_STRAIGHT')
            return self.output(0.0, 0.0, 'CHECK_PERSON_OUTSIDE_PATH')
        if self.phase == 'CHECK_EXIT':
            self.phase = 'TURN'
            self.clear_count = 0
            self.clear_started = None

        if left < MIN_GAP and right < MIN_GAP:
            self.phase = 'WAIT_GAP'
            self.gap_open_count = 0
            self.turn = 0.0
            self.clear_started = None
            self.clear_count = 0
            return self.output(0.0, 0.0, 'WAIT_FOR_VISUAL_GAP')
        if self.phase == 'CONFIRM':
            # Inherited corrected follower mapping: image-left=-, image-right=+.
            # User confirmed physical turning matches these labels.
            occupied_center = (min(p.left for p in central)
                               + max(p.right for p in central)) / 2
            direction = 1 if occupied_center < 0.45 else -1
            if left < MIN_GAP or right - left > 0.05:
                direction = 1
            elif right < MIN_GAP or left - right > 0.05:
                direction = -1
            self.confirm_count = self.confirm_count + 1 if direction == self.direction else 1
            self.direction = direction
            if self.confirm_count < 3 or now - self.encountered < STOP_DWELL:
                return self.output(0.0, 0.0, 'CONFIRM_PERSON_STOP')
            self.phase = 'TURN'
            self.turn_started = now
        if (left if self.direction < 0 else right) < MIN_GAP:
            # Do not turn into an apparently blocked side. A future set of
            # valid frames may choose a different direction after confirmation.
            self.phase = 'WAIT_GAP'
            self.gap_open_count = 0
            self.turn = 0.0
            self.clear_started = None
            self.clear_count = 0
            return self.output(0.0, 0.0, 'CHOSEN_GAP_CLOSED_WAIT')
        self.turn = 0.65 * self.turn + 0.35 * TURN * self.direction
        return self.output(0.0, self.turn,
                           'TURN_AWAY_LEFT' if self.direction < 0 else 'TURN_AWAY_RIGHT')

    def start_avoidance(self, central, now):
        self.phase = 'CONFIRM'
        self.encountered = now
        self.tracked_boxes = list(central)
        self.turn_started = None
        self.exit_started = None
        self.clear_started = None
        self.clear_count = 0
        self.confirm_count = 0
        self.direction = 0
        self.turn = 0.0
        self.missing_since = None
        self.gap_open_count = 0
        return self.output(0.0, 0.0, 'PERSON_IN_PATH_STOP')

    def person_on_expected_side(self, boxes):
        if self.direction > 0:  # Turn right, keep encountered person to the left.
            return all(p.right + PADDING < PATH_LEFT for p in boxes)
        return all(p.left - PADDING > PATH_RIGHT for p in boxes)



def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--publish-preview', action='store_true',
                        help='Publish proposed Twist commands on the separate preview topic.')
    parser.add_argument('--self-test', action='store_true', help='Run logic tests without ROS.')
    args = parser.parse_args()
    if args.self_test:
        run_tests()
        return
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import String
    from geometry_msgs.msg import Twist

    class PersonAvoidDemo(Node):
        def __init__(self):
            super().__init__('person_avoid_demo')
            self.controller = Controller()
            self.cmd_pub = (self.create_publisher(Twist, '/camera_person_avoid/preview_cmd', 10)
                            if args.publish_preview else None)
            self.state_pub = self.create_publisher(String, '/camera_person_avoid/state', 10)
            self.create_subscription(String, '/camera_observer/detections', self.on_frame,
                                     qos_profile_sensor_data)
            self.last_state = None
            self.last_log = 0.0
            self.create_timer(0.05, self.tick)
            mode = ('PUBLISHING /camera_person_avoid/preview_cmd'
                    if self.cmd_pub else 'LOG ONLY: no movement commands published')
            self.get_logger().info('PERSON AVOID DEMO V7 | ' + mode)

        def on_frame(self, msg):
            self.controller.receive(msg.data, time.monotonic())
            self.tick()  # React to a person/fault immediately, not only at next timer.

        def tick(self):
            now = time.monotonic()
            v, w, state = self.controller.step(now)
            if self.cmd_pub:
                cmd = Twist()
                cmd.linear.x, cmd.angular.z = v, w
                self.cmd_pub.publish(cmd)
            self.state_pub.publish(String(data=state))
            if state != self.last_state or now - self.last_log >= 1.0:
                self.get_logger().info(f'{state} | proposed_forward={v:.3f} | '
                                       f'proposed_turn={w:.3f} | {self.controller.summary}')
                self.last_state, self.last_log = state, now

    rclpy.init()
    node = PersonAvoidDemo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok() and node.cmd_pub:
            node.cmd_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def run_tests():
    import unittest

    def person(x=0.5, confidence=0.9, width=0.18):
        return {'class_id': 0, 'confidence': confidence, 'area_fraction': width * 0.6,
                'xyxy_px': [(x-width/2)*1000, 200, (x+width/2)*1000, 800]}

    class Checks(unittest.TestCase):
        def setUp(self):
            self.c = Controller()
            self.now, self.seq = 100.0, 0

        def feed(self, boxes=(), **changes):
            self.now += 0.1
            self.seq += 1
            obs = {'schema': 'ugv.camera.observation.v1', 'inference_enabled': True,
                   'capture_received_monotonic_ns': int(self.now*1e9),
                   'sequence': self.seq, 'session_id': 'test',
                   'image': {'width': 1000, 'height': 1000}, 'detections': list(boxes)}
            obs.update(changes)
            self.last = json.dumps(obs)
            self.c.receive(self.last, self.now)
            return self.c.step(self.now)

        def turning(self, x=0.4):
            for _ in range(9):
                out = self.feed([person(x)])
            self.assertEqual(self.c.phase, 'TURN')
            return out

        def exit_move(self, right=True):
            self.turning(0.4 if right else 0.6)
            xs = [0.35,0.3,0.25,0.2,0.14] if right else [0.65,0.7,0.75,0.8,0.86]
            for x in xs:
                self.feed([person(x)])
            for _ in range(7):
                out = self.feed([person(xs[-1])])
            self.assertEqual(self.c.phase, 'EXIT')
            self.assertEqual(out[:2], (FORWARD, 0.0))

        def test_wait_camera(self):
            self.assertEqual(self.c.step(self.now)[:2], (0.0,0.0))

        def test_startup_then_straight(self):
            for _ in range(4):
                self.assertEqual(self.feed()[:2], (0.0,0.0))
            self.assertEqual(self.feed()[:2], (FORWARD,0.0))

        def test_first_person_stops_immediately(self):
            for _ in range(5): self.feed()
            self.assertEqual(self.feed([person()])[:2], (0.0,0.0))

        def test_person_left_turn_right(self):
            self.assertGreater(self.turning()[1], 0)

        def test_person_right_turn_left(self):
            self.assertLess(self.turning(0.6)[1], 0)

        def test_center_nonzero(self):
            self.assertNotEqual(self.turning(0.5)[1], 0)

        def test_full_right_sequence(self):
            self.exit_move()
            for _ in range(35): out = self.feed([person(0.14)])
            self.assertEqual(out[:2], (FORWARD,0.0))
            self.assertEqual(out[2], 'CONTINUE_STRAIGHT')

        def test_full_left_sequence(self):
            self.exit_move(False)
            for _ in range(35): out = self.feed([person(0.86)])
            self.assertEqual(out[2], 'CONTINUE_STRAIGHT')

        def test_exit_does_not_use_turn_deadline(self):
            self.turning()
            for _ in range(40): self.feed([person(0.4)])
            for x in (0.35,0.3,0.25,0.2,0.14): self.feed([person(x)])
            for _ in range(7): self.feed([person(0.14)])
            self.assertEqual(self.c.phase, 'EXIT')
            for _ in range(10): self.feed([person(0.14)])
            self.assertEqual(self.c.phase, 'EXIT')

        def test_no_forward_while_turning(self):
            for _ in range(20):
                self.assertEqual(self.feed([person()])[0], 0)

        def test_split_and_merge(self):
            self.turning()
            for boxes in ([person(0.35,width=0.12),person(0.45,width=0.12)],
                          [person(0.4)], [person(0.4),person(0.4)]):
                out = self.feed(boxes)
                self.assertGreater(out[1], 0)
                self.assertEqual(out[0], 0)

        def test_unrelated_person_cannot_replace_lost_person(self):
            self.turning()
            out = self.feed([person(0.86)])
            self.assertEqual(out[:2], (0.0,0.0))
            self.assertIn('LOST',out[2])

        def test_one_of_multiple_encountered_people_disappears(self):
            for _ in range(9): self.feed([person(0.27,width=0.12),person(0.48,width=0.12)])
            out = self.feed([person(0.27,width=0.12)])
            self.assertEqual(out[:2], (0.0,0.0))
            self.assertIn('LOST',out[2])

        def test_person_lost_during_turn(self):
            self.turning()
            self.assertEqual(self.feed()[:2], (0.0,0.0))
            self.assertEqual(self.feed()[:2], (0.0,0.0))

        def test_person_lost_after_exit_continues(self):
            self.exit_move()
            started = self.c.exit_started
            self.assertEqual(self.feed()[:2], (FORWARD,0.0))
            for _ in range(35): out = self.feed()
            self.assertEqual(out[:2], (FORWARD,0.0))
            self.assertEqual(out[2], 'CONTINUE_STRAIGHT')
            self.assertEqual(self.c.exit_started, started)

        def test_side_detection_flicker_does_not_stop_cruise(self):
            self.exit_move()
            started = self.c.exit_started
            for i in range(35):
                out = self.feed([person(0.14)] if i % 2 else [])
            self.assertEqual(self.c.exit_started, started)
            self.assertEqual(out[:2], (FORWARD,0.0))
            self.assertEqual(out[2], 'CONTINUE_STRAIGHT')

        def test_camera_loss_during_empty_departure_stops(self):
            self.exit_move()
            self.feed()
            self.assertEqual(self.c.step(self.now+0.51)[:2], (0.0,0.0))
            self.assertIn('CAMERA_TIMEOUT', self.c.command[2])

        def test_person_lost_during_recheck_never_starts_departure(self):
            self.turning()
            for x in (0.35,0.3,0.25,0.2,0.14): self.feed([person(x)])
            self.assertEqual(self.c.phase, 'CHECK_EXIT')
            out = self.feed()
            self.assertEqual(out[:2], (0.0,0.0))
            self.assertIsNone(self.c.exit_started)

        def test_person_returns_after_absence_in_departure(self):
            self.exit_move()
            self.feed()
            self.assertEqual(self.feed([person(0.5)])[:2], (0.0,0.0))

        def test_person_reenters_during_exit(self):
            self.exit_move()
            out = self.feed([person(0.2)])
            self.assertEqual(out[:2], (0.0,0.0))
            self.assertEqual(out[2], 'PERSON_IN_PATH_STOP')

        def test_new_person_blocks_exit(self):
            self.exit_move()
            out = self.feed([person(0.14),person(0.5)])
            self.assertEqual(out[:2], (0.0,0.0))

        def test_clear_check_duplicate_frames(self):
            self.turning()
            for x in (0.35,0.3,0.25,0.2,0.14): self.feed([person(x)])
            for _ in range(10):
                self.c.receive(self.last,self.now)
                self.assertEqual(self.c.step(self.now)[:2], (0.0,0.0))
            self.assertEqual(self.c.clear_count,1)

        def test_clear_check_reentry_restarts_confirmation(self):
            self.turning()
            for x in (0.35,0.3,0.25,0.2,0.14): self.feed([person(x)])
            self.feed([person(0.2)])
            self.assertEqual(self.c.phase,'TURN')
            self.assertEqual(self.c.clear_count,0)

        def test_wrong_side_never_resumes(self):
            self.turning()
            for x in (0.5,0.6,0.7,0.8,0.86): out=self.feed([person(x)])
            self.assertEqual(out[:2], (0.0,0.0))
            self.assertNotEqual(self.c.phase,'EXIT')

        def test_lower_confidence_can_turn(self):
            for _ in range(9): out=self.feed([person(0.4,confidence=0.45)])
            self.assertGreater(out[1],0)

        def test_large_box_can_turn_if_gap_available(self):
            for _ in range(9): out=self.feed([person(0.35,width=0.45)])
            self.assertGreater(out[1],0)

        def test_blocked_sides_stop(self):
            for _ in range(9): out=self.feed([person(0.27,width=0.3),person(0.73,width=0.3)])
            self.assertEqual(out[:2],(0.0,0.0))
            self.assertIn('GAP',out[2])

        def test_blocked_gap_rechecks_and_recovers(self):
            # A blocked image must remain stopped, not permanently latch.
            blocked = [person(0.27,width=0.3),person(0.73,width=0.3)]
            for _ in range(30): out = self.feed(blocked)
            self.assertEqual(self.c.phase, 'WAIT_GAP')
            self.assertIsNone(self.c.halt_reason)
            self.assertEqual(out[:2], (0.0, 0.0))
            # Stable opening permits a new confirmation and turn.
            for _ in range(12): out = self.feed([person(0.4)])
            self.assertEqual(self.c.phase, 'TURN')
            self.assertNotEqual(out[1], 0.0)

        def test_single_missing_detection_stops_then_recovers(self):
            self.turning()
            out = self.feed()
            self.assertEqual(out[:2], (0.0, 0.0))
            self.assertEqual(out[2], 'WAIT_PERSON_REDETECTION')
            self.assertIsNone(self.c.halt_reason)
            out = self.feed([person(0.4)])
            self.assertEqual(self.c.phase, 'TURN')
            self.assertNotEqual(out[1], 0.0)

        def test_recheck_dropout_resets_confirmation_cleanly(self):
            self.turning()
            for x in (0.35, 0.3, 0.25, 0.2, 0.14):
                self.feed([person(x)])
            self.assertEqual(self.c.phase, 'CHECK_EXIT')
            out = self.feed()
            self.assertEqual(out[:2], (0.0, 0.0))
            self.assertEqual(self.c.phase, 'TURN')
            for _ in range(8):
                out = self.feed([person(0.14)])
            self.assertEqual(self.c.phase, 'EXIT')
            self.assertEqual(out[:2], (FORWARD, 0.0))

        def test_persistent_loss_still_fails_safe(self):
            self.turning()
            for _ in range(10): out = self.feed()
            self.assertEqual(out[:2], (0.0, 0.0))
            self.assertIn('PERSON_LOST', out[2])
            self.assertEqual(self.c.phase, 'HALTED')

        def test_turn_timeout(self):
            for _ in range(80): out=self.feed([person()])
            self.assertEqual(out[:2],(0.0,0.0))
            self.assertIn('TURN_TIME_LIMIT',out[2])

        def test_straight_keeps_going(self):
            for _ in range(1000): out=self.feed()
            self.assertEqual(out[:2],(FORWARD,0.0))
            self.assertEqual(out[2],'DRIVE_STRAIGHT')

        def test_second_avoidance_resets_previous_turn(self):
            self.exit_move()
            for _ in range(100): self.feed()
            out = self.feed([person(0.6)])
            self.assertEqual(out[:2], (0.0,0.0))
            self.assertIsNone(self.c.turn_started)
            self.assertEqual(self.c.direction, 0)
            for _ in range(9): out = self.feed([person(0.6)])
            self.assertLess(out[1], 0)
            self.assertEqual(out[0], 0)

        def test_reentry_stops_then_replans_without_restart(self):
            self.exit_move()
            self.assertEqual(self.feed([person(0.4)])[:2],(0.0,0.0))
            for _ in range(9): out=self.feed([person(0.4)])
            self.assertGreater(out[1],0)
            self.assertEqual(out[0],0)

        def test_long_continuation_still_times_out_camera(self):
            self.exit_move()
            for _ in range(1000): out=self.feed()
            self.assertEqual(out[:2],(FORWARD,0.0))
            self.assertEqual(self.c.step(self.now+0.51)[:2],(0.0,0.0))

        def test_stale_stops_in_every_motion_phase(self):
            for phase in ('STRAIGHT','TURN','EXIT'):
                self.setUp()
                if phase=='STRAIGHT':
                    for _ in range(5): self.feed()
                elif phase=='TURN': self.turning()
                else: self.exit_move()
                self.assertEqual(self.c.step(self.now+0.51)[:2],(0.0,0.0))
                self.assertEqual(self.feed([person(0.14)])[:2],(0.0,0.0))

        def test_session_restart_stops(self):
            self.turning()
            self.assertEqual(self.feed([person()],session_id='new')[:2],(0.0,0.0))

        def test_invalid_data(self):
            for raw in ('null','broken','{}','[]'):
                c=Controller();c.receive(raw,self.now)
                self.assertEqual(c.step(self.now)[:2],(0.0,0.0))
                self.assertIsNotNone(c.halt_reason)

        def test_inference_disabled(self):
            self.turning()
            self.assertEqual(self.feed(inference_enabled=False)[:2],(0.0,0.0))

        def test_nonfinite_box(self):
            b=person();b['xyxy_px'][0]=float('nan')
            self.assertEqual(self.feed([b])[:2],(0.0,0.0))
            self.assertIsNotNone(self.c.halt_reason)

    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    if not result.wasSuccessful(): raise SystemExit(1)


if __name__ == '__main__':
    main()

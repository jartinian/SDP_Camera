import ast
import importlib
import json
from pathlib import Path
from queue import Empty
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from ugv_camera.core import CameraConfig, detection, observation, uart_packet, parse_uart_packet
from ugv_camera.cli import parser, validate, infer, run
from ugv_camera.capture import Capture
from ugv_camera.outputs import FrameHub, Recorder, StreamServer, UartObserver


def sample():
    return observation('test', 0, time.time_ns(), time.monotonic_ns(), 1280, 720,
                       [detection(11, 'stop sign', .9, [500, 200, 780, 500], 1280, 720)])


class CoreTests(unittest.TestCase):
    def test_pipeline(self):
        value = CameraConfig().pipeline()
        for token in ('sensor-id=0', 'sensor-mode=2', 'width=(int)1920', 'height=(int)1080',
                      'framerate=(fraction)60/1', 'format=(string)BGR', 'max-buffers=1'):
            self.assertIn(token, value)
        self.assertNotIn('enc', value)

    def test_bad_camera_config(self):
        for kwargs in ({'width': 0}, {'width': 1279}, {'sensor_id': -1}, {'flip': 8}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CameraConfig(**kwargs).pipeline()

    def test_normalized_actual_resolution(self):
        d = sample()['detections'][0]
        self.assertEqual(d['center_normalized'][0], .5)
        self.assertAlmostEqual(d['area_fraction'], 280 * 300 / (1280 * 720))
        self.assertNotIn('angle', d)
        self.assertNotIn('distance', d)

    def test_reject_invalid_boxes(self):
        for coords in ([float('nan'), 0, 2, 2], [5, 0, 3, 2], [-4, 0, -1, 2], [1, 2]):
            with self.subTest(coords=coords), self.assertRaises(ValueError):
                detection(0, 'person', .5, coords, 640, 480)
        with self.assertRaises(ValueError):
            detection(0, 'person', float('nan'), [0, 0, 2, 2], 640, 480)

    def test_empty_and_camera_only_distinct(self):
        args = ('test', 1, 1, 1, 1280, 720, [])
        self.assertEqual(observation(*args)['state'], 'no_detections')
        self.assertEqual(observation(*args, inference_enabled=False)['state'], 'camera_only')

    def test_uart_roundtrip_and_corruption(self):
        packet = uart_packet(sample())
        self.assertEqual(parse_uart_packet(packet)['detections'][0]['class_id'], 11)
        self.assertLessEqual(len(packet), 2048)
        with self.assertRaises(ValueError):
            parse_uart_packet(packet.replace(b'stop sign', b'wrongword'))
        for broken in (b'NO_TARGET\n', b'R_\n', b'x' * 2049):
            with self.assertRaises(ValueError):
                parse_uart_packet(broken)

    def test_safe_defaults_and_validation(self):
        a = parser().parse_args([])
        validate(a)
        self.assertFalse(a.ros or a.stream or a.show or a.uart_port)
        self.assertEqual(a.record, 'off')
        self.assertEqual(a.device, '0')
        for flags in (['--uart-port', '/dev/FAKE'], ['--uart-observer-confirmed'],
                      ['--ros-images'], ['--no-inference', '--track'], ['--conf', 'nan'],
                      ['--record-required'], ['--uart-rate', '6'], ['--frames', '-1']):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                validate(parser().parse_args(flags))

    def test_no_device_side_effect_on_import(self):
        with patch('serial.Serial', side_effect=AssertionError('opened UART')):
            importlib.reload(importlib.import_module('ugv_camera.outputs'))

    def test_control_boundary(self):
        for path in (Path(__file__).parents[1] / 'ugv_camera').glob('*.py'):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    self.assertNotIn('geometry_msgs', ast.unparse(node))
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    self.assertFalse(node.value.startswith('/cmd_vel'))
                    self.assertFalse(node.value.startswith('/vision/'))
        source = (Path(__file__).parents[1] / 'ugv_camera/outputs.py').read_text()
        self.assertIn('use_global_arguments=False', source)

    def test_one_inference_and_persistent_tracking(self):
        import numpy as np
        model = Mock()
        result = Mock(boxes=None)
        model.predict.return_value = [result]
        model.track.return_value = [result]
        a = parser().parse_args([])
        infer(model, np.zeros((2, 2, 3)), a)
        model.predict.assert_called_once()
        model.track.assert_not_called()
        a.track = True
        infer(model, np.zeros((2, 2, 3)), a)
        self.assertTrue(model.track.call_args.kwargs['persist'])
        self.assertEqual(model.track.call_args.kwargs['tracker'], 'bytetrack.yaml')


class OutputTests(unittest.TestCase):
    def test_shared_latest_frame_and_staleness(self):
        hub = FrameHub(.02)
        hub.publish(b'one')
        hub.publish(b'two')
        self.assertEqual(hub.wait(-1), (2, b'two'))
        self.assertEqual(hub.wait(-1), (2, b'two'))
        time.sleep(.03)
        self.assertIsNone(hub.wait(-1))
        hub.close()
        self.assertIsNone(hub.wait(-1))

    def test_waiter_wakes_on_close(self):
        hub = FrameHub(10)
        result = []
        t = threading.Thread(target=lambda: result.append(hub.wait(0)))
        t.start()
        hub.close()
        t.join(1)
        self.assertFalse(t.is_alive())
        self.assertEqual(result, [None])

    def test_uart_partial_write_fails_and_rate_limits(self):
        uart = UartObserver.__new__(UartObserver)
        uart.serial = Mock(out_waiting=0)
        uart.period = .5
        uart.last = time.monotonic()
        uart.publish(sample())
        uart.serial.write.assert_not_called()
        uart.last = -float('inf')
        uart.serial.write.return_value = 1
        with self.assertRaises(RuntimeError):
            uart.publish(sample())

    def test_software_recording_roundtrip(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            f = np.full((48, 64, 3), 100, dtype=np.uint8)
            writer = Recorder(tmp, f, 10, 'avi')
            for i in range(5):
                obs = sample()
                obs['sequence'] = i
                writer.write(f, obs)
            writer.close()
            cap = cv2.VideoCapture(str(writer.path), cv2.CAP_FFMPEG)
            count = 0
            while cap.read()[0]:
                count += 1
            cap.release()
            self.assertEqual(count, 5)
            self.assertEqual(len((Path(tmp) / 'video_timestamps.csv').read_text().splitlines()), 6)

    def test_mp4_fallback(self):
        import cv2
        import numpy as np
        real = cv2.VideoWriter
        def writer(path, *args):
            if str(path).endswith('.mp4'):
                return Mock(isOpened=lambda: False)
            return real(path, *args)
        with tempfile.TemporaryDirectory() as tmp, patch('cv2.VideoWriter', side_effect=writer):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            r = Recorder(tmp, frame, mode='auto')
            self.assertEqual(r.path.suffix, '.avi')
            r.write(frame, sample())
            r.close()

    def test_http_stream_snapshot_and_shutdown(self):
        from urllib.request import urlopen
        try:
            server = StreamServer('127.0.0.1', 0)
        except PermissionError:
            self.skipTest('Sandbox blocks local network sockets')
        try:
            server.hub.publish(b'JPEG_TEST')
            root = f'http://127.0.0.1:{server.server.server_port}'
            with urlopen(root + '/snapshot.jpg', timeout=2) as response:
                self.assertEqual(response.read(), b'JPEG_TEST')
            with urlopen(root + '/video', timeout=2) as response:
                self.assertIn(b'--frame', response.readline())
        finally:
            server.close()
        self.assertFalse(server.thread.is_alive())


class LifecycleTests(unittest.TestCase):
    def test_capture_timeout_is_bounded(self):
        cap = Capture.__new__(Capture)
        cap.frames = Mock()
        cap.frames.get.side_effect = Empty()
        cap.events = Mock()
        cap.events.get_nowait.side_effect = Empty()
        cap.process = Mock(is_alive=lambda: True)
        before = time.monotonic()
        with self.assertRaises(TimeoutError):
            cap.read(.02)
        self.assertLess(time.monotonic() - before, .5)

    def test_cleanup_on_inference_fault(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            a = parser().parse_args(['--output', tmp, '--frames', '1'])
            cap = Mock()
            cap.read.return_value = (time.time_ns(), time.monotonic_ns(), np.zeros((48, 64, 3), dtype=np.uint8))
            with patch('ugv_camera.cli.load_model', return_value=Mock()), \
                 patch('ugv_camera.capture.Capture', return_value=cap), \
                 patch('ugv_camera.cli.infer', side_effect=RuntimeError('injected inference failure')):
                self.assertEqual(run(a), 1)
            cap.close.assert_called_once()
            summary = json.loads(next(Path(tmp).glob('*/summary.json')).read_text())
            self.assertEqual(summary['state'], 'fault')
            self.assertIn('injected', summary['error'])

    def test_stale_frame_not_published(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            a = parser().parse_args(['--output', tmp, '--frames', '1', '--no-inference'])
            cap = Mock()
            cap.read.return_value = (time.time_ns(), time.monotonic_ns() - 4_000_000_000,
                                     np.zeros((48, 64, 3), dtype=np.uint8))
            with patch('ugv_camera.capture.Capture', return_value=cap):
                self.assertEqual(run(a), 1)
            self.assertEqual(next(Path(tmp).glob('*/observations.jsonl')).read_text(), '')
            cap.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()

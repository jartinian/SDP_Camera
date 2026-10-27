"""Bounded latest-frame acquisition, isolated from potentially blocking native reads."""

import multiprocessing as mp
from queue import Empty, Full
import time


def _worker(config, source_file, frames, events, stop):
    import cv2

    cap = None
    # Do not wait forever for a queue feeder if the parent has stopped consuming.
    frames.cancel_join_thread()
    try:
        if source_file:
            cap = cv2.VideoCapture(source_file, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(config.pipeline(), cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            raise RuntimeError("Cannot open capture; check GStreamer, camera ownership and Argus")
        fps = cap.get(cv2.CAP_PROP_FPS) if source_file else 0
        period = 1.0 / fps if source_file and 0 < fps < 1000 else 1.0 / 30
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                events.put(("eof" if source_file else "error", "Capture returned no frame"))
                break
            item = (time.time_ns(), time.monotonic_ns(), frame)
            try:
                frames.put_nowait(item)
            except Full:
                try:
                    frames.get_nowait()
                except Empty:
                    pass
                try:
                    frames.put_nowait(item)
                except Full:
                    pass
            if source_file:
                stop.wait(period)
    except Exception as exc:
        events.put(("error", str(exc)))
    finally:
        if cap is not None:
            cap.release()


class Capture:
    def __init__(self, config, source_file=None):
        ctx = mp.get_context("spawn")
        self.frames = ctx.Queue(maxsize=1)
        self.events = ctx.Queue()
        self.stop = ctx.Event()
        self.process = ctx.Process(target=_worker,
                                   args=(config, source_file, self.frames, self.events, self.stop),
                                   daemon=True)
        self.process.start()

    def read(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                return self.frames.get(timeout=min(0.1, max(0.001, deadline - time.monotonic())))
            except Empty:
                try:
                    kind, message = self.events.get_nowait()
                except Empty:
                    if not self.process.is_alive():
                        raise RuntimeError("Capture process exited unexpectedly")
                    continue
                if kind == "eof":
                    raise EOFError(message)
                raise RuntimeError(message)
        raise TimeoutError(f"No camera frame within {timeout:g} seconds")

    def close(self):
        self.stop.set()
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=1)
        self.frames.close()
        self.events.close()

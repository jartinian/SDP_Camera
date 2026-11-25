
#!/usr/bin/env python3
"""
Lane Tracking V2 - White Tape Detection and Curve Tracking

Camera-only visual testing.
NO MOTOR COMMANDS, ROS COMMANDS, OR VEHICLE MOVEMENT.
"""

import cv2
import numpy as np


class LaneTracker:
    def __init__(self):
        self.previous_curve = None
        self.smoothed_target = None
        self.smoothed_target_y = None
        self.missing_frames = 10

    def process(self, frame):
        h, w = frame.shape[:2]

        # Work at half resolution for faster processing.
        reduced = cv2.resize(
            frame,
            (w // 2, h // 2),
            interpolation=cv2.INTER_AREA
        )

        rh, rw = reduced.shape[:2]

        hsv = cv2.cvtColor(reduced, cv2.COLOR_BGR2HSV)
        bright = hsv[:, :, 2].astype(np.float32)

        # Check the approximate floor brightness.
        floor_region = bright[
            int(rh * 0.70):int(rh * 0.95),
            int(rw * 0.20):int(rw * 0.80)
        ]

        floor_level = float(np.median(floor_region))
        floor_ok = floor_level >= 118

        # --------------------------------------------------
        # WHITE TAPE DETECTION
        # Retain the filtering that worked in V1.
        # --------------------------------------------------

        b, g, r = cv2.split(reduced.astype(np.float32))

        chroma = (
            0.8 * (b - r)
            + 0.4 * (g - r)
        )

        bg_bright = cv2.GaussianBlur(
            bright, (0, 0), 13
        )

        bg_chroma = cv2.GaussianBlur(
            chroma, (0, 0), 13
        )

        score = (
            0.5 * (bright - bg_bright)
            + 1.3 * (chroma - bg_chroma)
        )

        # Detect saturated blue tape using HSV color.
        # Initial thresholds; calibrate with the live mask.
        mask = cv2.inRange(
            hsv,
            (95, 65, 45),
            (135, 255, 255)
        )

        # Ignore the upper part of the camera view.
        mask[:int(rh * 0.42)] = 0

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((7, 3), np.uint8)
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8)
        )

        # --------------------------------------------------
        # VISUAL DISPLAY
        # --------------------------------------------------

        annotated = frame.copy()

        # Preferred target row, not a mandatory detection row.
        target_y = int(rh * 0.75)

        # Yellow vertical line = camera center.
        cv2.line(
            annotated,
            (w // 2, 0),
            (w // 2, h - 1),
            (0, 255, 255),
            2
        )

        # Cyan horizontal line = preferred look-ahead row.
        cv2.line(
            annotated,
            (0, target_y * 2),
            (w - 1, target_y * 2),
            (255, 200, 0),
            2
        )

        result = {
            "direction": "STOP - LINE LOST",
            "error": "",
            "target_x": "",
            "target_y": "",
            "floor_brightness": round(floor_level, 1),
            "confidence": "unvalidated",
            "note": "No continuous tape path",
            "lane_detected": False,
            "path_points": 0,
            "curve_indicator": 0.0
        }

        # --------------------------------------------------
        # MULTI-POINT TAPE TRACKING
        # --------------------------------------------------

        if floor_ok:

            # Scan horizontal rows from near to far.
            ys = list(range(
                int(rh * 0.94),
                int(rh * 0.43),
                -14
            ))

            rows = []

            for y in ys:

                strip = mask[
                    max(0, y - 6):min(rh, y + 7)
                ]

                counts = np.count_nonzero(
                    strip, axis=0
                ).astype(np.float32)

                counts = cv2.GaussianBlur(
                    counts.reshape(1, -1),
                    (5, 1),
                    0
                )[0]

                active = counts >= max(
                    2.0,
                    strip.shape[0] * 0.20
                )

                edges = np.diff(
                    np.pad(
                        active.astype(np.int8),
                        (1, 1)
                    )
                )

                starts = np.flatnonzero(edges == 1)
                ends = np.flatnonzero(edges == -1)

                centers = []

                for left, right in zip(starts, ends):

                    width = right - left

                    if 2 <= width <= 55:
                        center = (
                            left + right - 1
                        ) / 2.0

                        centers.append(center)

                rows.append(centers)

            # --------------------------------------------------
            # FIND CONTINUOUS TAPE PATHS
            # --------------------------------------------------

            best = None

            for start_i, centers in enumerate(rows):

                for start_x in centers:

                    points = [
                        (ys[start_i], start_x)
                    ]

                    last_x = start_x
                    last_step = 0.0
                    gaps = 0

                    for j in range(
                        start_i + 1,
                        len(ys)
                    ):

                        options = rows[j]

                        predicted = (
                            last_x
                            + last_step * (gaps + 1)
                        )

                        if not options:

                            gaps += 1

                        else:

                            next_x = min(
                                options,
                                key=lambda x: abs(
                                    x - predicted
                                )
                            )

                            distance = abs(
                                next_x - predicted
                            )

                            if distance <= max(
                                30,
                                rw * 0.10
                            ):

                                step = (
                                    next_x - last_x
                                ) / (gaps + 1)

                                last_step = (
                                    0.5 * last_step
                                    + 0.5 * step
                                )

                                last_x = next_x

                                points.append(
                                    (ys[j], next_x)
                                )

                                gaps = 0

                            else:

                                gaps += 1

                        if gaps > 2:
                            break

                    # Require multiple tape measurements.
                    if len(points) < 5:
                        continue

                    vertical_span = (
                        points[0][0]
                        - points[-1][0]
                    )

                    if vertical_span < 55:
                        continue

                    py = np.array(
                        [p[0] for p in points],
                        dtype=float
                    )

                    px = np.array(
                        [p[1] for p in points],
                        dtype=float
                    )

                    # Fit a smooth path.
                    degree = (
                        2 if len(points) >= 7
                        else 1
                    )

                    fitted = np.polyfit(
                        py, px, degree
                    )

                    if degree == 1:
                        fitted = np.array([
                            0.0,
                            fitted[0],
                            fitted[1]
                        ])

                    residual = float(
                        np.median(
                            np.abs(
                                np.polyval(
                                    fitted, py
                                ) - px
                            )
                        )
                    )

                    if residual > 18:
                        continue

                    # ------------------------------------------
                    # V2 FIX: TARGET WITHIN THE VISIBLE TAPE
                    # ------------------------------------------

                    # If tape ends above the cyan reference,
                    # target the nearest visible point instead.
                    visible_target_y = int(
                        np.clip(
                            target_y,
                            np.min(py),
                            np.max(py)
                        )
                    )

                    target = float(
                        np.polyval(
                            fitted,
                            visible_target_y
                        )
                    )

                    if not (0 <= target < rw):
                        continue

                    # ------------------------------------------
                    # TRACK THE SAME TAPE BETWEEN FRAMES
                    # ------------------------------------------

                    history_penalty = 0.0

                    if (
                        self.previous_curve is not None
                        and self.missing_frames <= 3
                    ):

                        previous_target = float(
                            np.polyval(
                                self.previous_curve,
                                visible_target_y
                            )
                        )

                        jump = abs(
                            target - previous_target
                        )

                        # Reject sudden switches to another line.
                        if jump > rw * 0.20:
                            continue

                        history_penalty = jump * 0.20

                    # Prefer longer, smoother paths starting
                    # closer to the vehicle.
                    quality = (
                        len(points) * 9
                        + vertical_span * 0.08
                        - start_i * 2.5
                        - residual * 0.5
                        - abs(start_x - rw / 2) * 0.015
                        - history_penalty
                    )

                    if (
                        best is None
                        or quality > best[0]
                    ):

                        best = (
                            quality,
                            fitted,
                            points,
                            target,
                            visible_target_y
                        )

            # --------------------------------------------------
            # DISPLAY THE BEST DETECTED PATH
            # --------------------------------------------------

            if best is not None:

                (
                    _,
                    curve,
                    points,
                    target,
                    visible_target_y
                ) = best

                # Smooth target movement when tracking
                # approximately the same image row.
                if (
                    self.smoothed_target is None
                    or self.smoothed_target_y is None
                    or self.missing_frames > 3
                    or abs(
                        visible_target_y
                        - self.smoothed_target_y
                    ) > 14
                ):

                    self.smoothed_target = target

                else:

                    self.smoothed_target = (
                        0.6 * self.smoothed_target
                        + 0.4 * target
                    )

                self.smoothed_target_y = visible_target_y
                self.previous_curve = curve
                self.missing_frames = 0

                offset = (
                    self.smoothed_target - rw / 2
                ) / (rw / 2)

                direction = (
                    "LEFT" if offset < -0.08
                    else "RIGHT" if offset > 0.08
                    else "STRAIGHT"
                )

                # Estimate bending within the visible path.
                measured_ys = np.array(
                    [p[0] for p in points],
                    dtype=float
                )

                samples = np.linspace(
                    np.min(measured_ys),
                    np.max(measured_ys),
                    3
                )

                curve_x = np.polyval(
                    curve, samples
                )

                bend = float(
                    curve_x[0]
                    - 2 * curve_x[1]
                    + curve_x[2]
                )

                # Draw fitted path only where tape was measured.
                top_y = int(
                    np.min(measured_ys)
                )

                bottom_y = int(
                    np.max(measured_ys)
                )

                path = []

                for y in range(
                    top_y,
                    bottom_y + 1,
                    3
                ):

                    x = int(
                        np.polyval(curve, y)
                    )

                    if 0 <= x < rw:

                        path.append(
                            (x * 2, y * 2)
                        )

                # Green = fitted tape trajectory.
                if len(path) >= 2:

                    cv2.polylines(
                        annotated,
                        [
                            np.array(
                                path,
                                dtype=np.int32
                            )
                        ],
                        False,
                        (0, 255, 0),
                        5
                    )

                # Purple = individual tracking points.
                for y, x in points:

                    cv2.circle(
                        annotated,
                        (
                            int(x * 2),
                            int(y * 2)
                        ),
                        5,
                        (255, 0, 255),
                        -1
                    )

                # Red = selected target on visible tape.
                cv2.circle(
                    annotated,
                    (
                        int(self.smoothed_target * 2),
                        visible_target_y * 2
                    ),
                    11,
                    (0, 0, 255),
                    -1
                )

                # Determine whether tape is ahead of reference.
                tape_ahead = (
                    visible_target_y < target_y - 10
                )

                note = (
                    "TAPE AHEAD - target on visible tape"
                    if tape_ahead
                    else "Tape crosses preferred target region"
                )

                result.update({
                    "direction": direction,
                    "error": round(float(offset), 3),
                    "target_x": int(
                        self.smoothed_target * 2
                    ),
                    "target_y": int(
                        visible_target_y * 2
                    ),
                    "lane_detected": True,
                    "path_points": len(points),
                    "curve_indicator": round(bend, 2),
                    "note": note
                })

        # --------------------------------------------------
        # HANDLE TEMPORARILY LOST TAPE
        # --------------------------------------------------

        if not result["lane_detected"]:

            self.missing_frames += 1

            if self.missing_frames > 3:

                self.previous_curve = None
                self.smoothed_target = None
                self.smoothed_target_y = None

        # --------------------------------------------------
        # STATUS TEXT
        # --------------------------------------------------

        if not result["lane_detected"]:
            shape_text = "NO TAPE PATH"

        elif abs(result["curve_indicator"]) > 10:
            shape_text = "CURVE CANDIDATE"

        else:
            shape_text = "STRAIGHT CANDIDATE"

        cv2.rectangle(
            annotated,
            (0, 0),
            (w, 105),
            (24, 24, 24),
            -1
        )

        cv2.putText(
            annotated,
            result["direction"],
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3
        )

        status = (
            f"Offset: {result['error']}   "
            f"Points: {result['path_points']}   "
            f"{shape_text}"
        )

        cv2.putText(
            annotated,
            status,
            (20, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (220, 220, 220),
            2
        )

        # Return debugging mask at original resolution.
        full_mask = cv2.resize(
            mask,
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

        return annotated, full_mask, result


# Persistent tracker for continuous live video.
_tracker = LaneTracker()


def process(frame):
    return _tracker.process(frame)

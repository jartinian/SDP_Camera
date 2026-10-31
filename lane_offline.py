#!/usr/bin/env python3
"""Offline white-tape line tracker. Safe: never sends motor commands.

Designed as a first prototype for the specific brightly lit, reflective-floor
1280x720 Jetson photos provided by the Senior Design team.

Usage:
    python lane_offline.py --input photos --output results
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def natural_order(path):
    import re
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', path.name)]


def process(frame):
    """Return annotated image, debug mask and measurement record.

    Steering is a *visual suggestion* only; not a validated motor command.
    """
    h, w = frame.shape[:2]
    reduced = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    rh, rw = reduced.shape[:2]
    hsv = cv2.cvtColor(reduced, cv2.COLOR_BGR2HSV)
    bright = hsv[:, :, 2].astype(np.float32)

    # Image-specific calibration: reject views of the ceiling, cables or
    # underexposed objects before trying to find the line.
    bottom_center = bright[int(rh * .70):int(rh * .95),
                           int(rw * .20):int(rw * .80)]
    floor_level = float(np.median(bottom_center))
    floor_ok = floor_level >= 118

    b, g, r = cv2.split(reduced.astype(np.float32))
    # White tape has a bluish/greenish cast in these camera frames.
    # Measure this cast RELATIVE to the nearby floor, plus local brightness.
    chroma = .8 * (b - r) + .4 * (g - r)
    bg_bright = cv2.GaussianBlur(bright, (0, 0), 13, 2)
    bg_chroma = cv2.GaussianBlur(chroma, (0, 0), 13, 2)
    score = .5 * (bright - bg_bright) + 1.3 * (chroma - bg_chroma)
    good = ((score > 9.0) & (bright > 80)).astype(np.uint8) * 255
    good[:int(rh * .42)] = 0
    good = cv2.morphologyEx(good, cv2.MORPH_CLOSE,
                            np.ones((7, 3), dtype=np.uint8))
    good = cv2.morphologyEx(good, cv2.MORPH_OPEN,
                            np.ones((3, 3), dtype=np.uint8))

    candidates = []
    if floor_ok:
        raw_lines = cv2.HoughLinesP(
            good, rho=1, theta=np.pi / 180, threshold=22,
            minLineLength=45, maxLineGap=30
        )
        if raw_lines is not None:
            for x1, y1, x2, y2 in raw_lines[:, 0]:
                dx = int(x2) - int(x1)
                dy = int(y2) - int(y1)
                length = float(np.hypot(dx, dy))
                if abs(dy) < 45 or abs(dy) < .40 * abs(dx):
                    continue
                if max(y1, y2) < .75 * rh or length < 60:
                    continue
                steps = np.linspace(0, 1, 31)
                xs = np.clip(np.rint(x1 + steps * dx).astype(int), 0, rw-1)
                ys = np.clip(np.rint(y1 + steps * dy).astype(int), 0, rh-1)
                line_bright = float(np.median(bright[ys, xs]))
                local_strength = float(np.median(score[ys, xs]))
                # Dark chair legs aren't white tape.
                if line_bright < 118 or local_strength < 5:
                    continue
                verticality = abs(dy) / max(length, 1)
                quality = length * (.35 + .65 * verticality)
                quality *= np.clip(.8 + .02 * local_strength, .8, 1.3)
                candidates.append((quality, (int(x1), int(y1),
                                             int(x2), int(y2)),
                                   line_bright, local_strength))

    annotated = frame.copy()
    y_target = int(.83 * h)
    cv2.line(annotated, (w // 2, 0), (w // 2, h-1), (0, 255, 255), 2)
    cv2.line(annotated, (0, y_target), (w - 1, y_target),
             (255, 200, 0), 2)
    cv2.line(annotated, (0, int(.42*h)), (w-1, int(.42*h)),
             (170, 140, 140), 1)

    result = dict(direction="STOP - LINE LOST", error="", target_x="",
                  floor_brightness=round(floor_level, 1),
                  confidence="unvalidated", note="No reliable candidate")
    if candidates:
        _, (x1, y1, x2, y2), _, strength = max(candidates, key=lambda p: p[0])
        X1, Y1, X2, Y2 = (x1*2, y1*2, x2*2, y2*2)
        # Extrapolate a detected straight segment to the steering reference row.
        if abs(Y2-Y1) > 1:
            target_x = X1 + (y_target - Y1)*(X2-X1)/(Y2-Y1)
            if 0 <= target_x < w:
                err = (target_x - w/2)/(w/2)
                direction = ("LEFT" if err < -.08 else
                             "RIGHT" if err > .08 else "STRAIGHT")
                cv2.line(annotated, (X1,Y1), (X2,Y2), (0,240,0), 6)
                cv2.circle(annotated, (int(target_x), y_target),
                           11, (0, 0, 255), -1)
                cv2.line(annotated, (w//2, y_target),
                         (int(target_x), y_target), (0, 200, 255), 3)
                result = dict(direction=direction, error=round(float(err), 3),
                              target_x=int(target_x),
                              floor_brightness=round(floor_level, 1),
                              confidence="unvalidated",
                              note="Segment fit; verify overlay visually")
            else:
                result["note"] = "Predicted line leaves image"

    title = result["direction"]
    # Filled banner for legibility on both reflective floors and dark objects.
    cv2.rectangle(annotated, (0,0), (w,90), (24,24,24), -1)
    cv2.putText(annotated, title, (20, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 1.05, (255, 255, 255), 3)
    err_str = str(result["error"])
    cv2.putText(annotated,
                f"Offset: {err_str}   Floor brightness: {floor_level:.0f}",
                (20,78), cv2.FONT_HERSHEY_SIMPLEX, .65, (220,220,220), 2)
    mask = cv2.resize(good, (w, h), interpolation=cv2.INTER_NEAREST)
    return annotated, mask, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('photos'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    if not args.input.is_dir():
        parser.error(f"Input folder does not exist: {args.input}")
    images = sorted((p for p in args.input.iterdir()
                     if p.suffix.lower() in ('.jpg','.jpeg','.png')),
                    key=natural_order)
    if not images:
        parser.error(f"No jpg/png images found in {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'results.csv').open('w', newline='', encoding='utf-8') as f:
        csvfile = csv.DictWriter(f, fieldnames=['image','direction','error',
                     'target_x','floor_brightness','confidence','note'])
        csvfile.writeheader()
        for p in images:
            img = cv2.imread(str(p))
            if img is None:
                print('Skipping unreadable image:', p.name)
                continue
            overlay, mask, row = process(img)
            cv2.imwrite(str(args.output / (p.stem + '_tracked.jpg')), overlay)
            cv2.imwrite(str(args.output / (p.stem + '_mask.png')), mask)
            csvfile.writerow(dict(image=p.name, **row))
            print(p.name, ':', row['direction'], 'offset=', row['error'])
    print('Finished:', args.output.resolve())


if __name__ == '__main__':
    main()

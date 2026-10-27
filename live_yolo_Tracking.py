#!/usr/bin/env python3
"""Compatibility launcher. All control behavior is removed; see docs/SCRIPT_REVIEW.md."""
import sys
from ugv_camera.cli import main

if __name__ == "__main__":
    print("Migrated launcher: observations only; UART and ROS disabled unless explicitly enabled.", file=sys.stderr)
    raise SystemExit(main(['--track'] + sys.argv[1:]))

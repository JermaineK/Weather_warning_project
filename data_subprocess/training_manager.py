#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training_manager.py

Thin delegator for training / prediction stages:
  - train-base            -> train_calibrate_eval.py
  - train-alert-specialist-> train_alert_specialist.py
  - predict-alerts        -> predict_and_alert.py
  - track-objects         -> features_subprocess/track_alert_objects.py (via relative path)
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict

HERE = Path(__file__).resolve().parent


ROUTING: Dict[str, Path] = {
    "train-base": HERE / "train_calibrate_eval.py",
    "base": HERE / "train_calibrate_eval.py",
    "train-alert-specialist": HERE / "train_alert_specialist.py",
    "specialist": HERE / "train_alert_specialist.py",
    "predict-alerts": HERE / "predict_and_alert.py",
    "predict": HERE / "predict_and_alert.py",
    "track-objects": HERE.parent / "features_subprocess" / "track_alert_objects.py",
    "track": HERE.parent / "features_subprocess" / "track_alert_objects.py",
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Training / prediction manager", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("mode", choices=sorted(ROUTING.keys()))
    ap.add_argument("--dry-run", action="store_true")
    args, extra = ap.parse_known_args()

    script = ROUTING[args.mode].resolve()
    if not script.exists():
        raise SystemExit(f"Script missing for mode {args.mode}: {script}")

    cmd = [sys.executable, str(script), *extra]
    print("$ " + " ".join(shlex.quote(c) for c in cmd))
    if args.dry_run:
        return 0
    rc = subprocess.run(cmd)
    return int(rc.returncode)


if __name__ == "__main__":
    sys.exit(main())

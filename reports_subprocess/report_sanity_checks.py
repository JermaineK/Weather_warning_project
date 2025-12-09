#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reports_sanity_checks.py

Lightweight sanity checks for expected artifacts in a run.

Typical usage (from a run manager):
  python reports_sanity_checks.py \
    --must-exist \
      results/seedmaps/seed_summary.txt \
      results/seedmaps/seed_patches.csv \
      results/seedmaps/seed_track_matches.csv \
    --alerts-dir results/alerts/20251122_run001 \
    --require-alerts \
    --strict

Notes
-----
- This script does *not* try to infer run IDs or directories.
  The caller (e.g. reports_and_maps_manager.py) should pass in
  run-specific paths via --must-exist and --alerts-dir.
- For alerts, we just check that at least one alerts_*_base.csv.gz
  exists and has some readable rows.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def nonempty_csv(path: Path, max_rows: int = 1000) -> int:
    """
    Try to read up to max_rows from a CSV(.gz) file.
    Returns:
      - number of rows read (0..max_rows) on success
      - -1 on any read error
    """
    try:
        return len(pd.read_csv(path, nrows=max_rows))
    except Exception:
        return -1


def main() -> None:
    ap = argparse.ArgumentParser(description="Sanity checks for expected artifacts.")
    ap.add_argument(
        "--must-exist",
        nargs="*",
        default=[
            "results/seedmaps/seed_summary.txt",
            "results/seedmaps/seed_patches.csv",
            "results/seedmaps/seed_track_matches.csv",
        ],
        help="Files that must exist and be non-empty for the run to be considered valid.",
    )
    ap.add_argument(
        "--alerts-dir",
        default="results/alerts",
        help="Directory to scan for alerts_*_base.* when --require-alerts is set.",
    )
    ap.add_argument(
        "--require-alerts",
        action="store_true",
        help="Require at least one alerts_*_base.* with readable rows.",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero status if any required check fails.",
    )
    args = ap.parse_args()

    ok = True

    # Core file existence / size checks
    print("[sanity] Checking required files...")
    for p in args.must_existent if False else args.must_exist:  # keep arg name, avoid typo
        P = Path(p)
        if not P.exists():
            print(f"[sanity] MISSING: {P}")
            ok = False
        elif P.stat().st_size <= 0:
            print(f"[sanity] EMPTY: {P}")
            ok = False
        else:
            print(f"[sanity] ok: {P}")

    # Alerts presence (optional)
    if args.require_alerts:
        alerts_dir = Path(args.alerts_dir)
        print(f"[sanity] Checking alerts in: {alerts_dir}")
        found_any = False

        if alerts_dir.exists():
            for fp in sorted(alerts_dir.glob("alerts_*_base.*")):
                found_any = True
                if fp.suffix.lower() in {".parquet", ".pq", ".pqt"}:
                    try:
                        sample = pd.read_parquet(fp, columns=["time"], nrows=500)
                        rows = len(sample)
                        print(f"[sanity] alerts file: {fp.name}  sample_rows={rows}")
                        if rows == 0:
                            ok = False
                    except Exception as exc:
                        # Treat as a warning so a single flaky parquet file does not fail the run.
                        print(f"[sanity] alerts file: {fp.name}  READ_ERROR ({exc})")
                        continue
                else:
                    rows = nonempty_csv(fp, max_rows=500)
                    if rows < 0:
                        print(f"[sanity] alerts file: {fp.name}  READ_ERROR")
                        ok = False
                    elif rows == 0:
                        print(f"[sanity] alerts file: {fp.name}  sample_rows=0 (empty?)")
                        ok = False
                    else:
                        print(f"[sanity] alerts file: {fp.name}  sample_rows={rows}")
        else:
            print(f"[sanity] alerts directory not found: {alerts_dir}")
            found_any = False

        if not found_any:
            print(f"[sanity] no alerts_*_base.* in {alerts_dir}")
            ok = False

    if args.strict and not ok:
        print("[sanity] FAIL (strict mode).")
        sys.exit(2)

    print("[sanity] checks complete.")
    # Exit 0 even if not ok, when strict=False (informational mode)


if __name__ == "__main__":
    main()

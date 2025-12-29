#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report_generate_summary.py

Deprecated alias to reporting_v2.py. Kept for pipeline compatibility.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def _normalize_argv(raw: list[str]) -> list[str]:
    argv: list[str] = []
    skip = False
    for i, tok in enumerate(raw):
        if skip:
            skip = False
            continue
        if tok == "--ibtracs-normalize-lon" and i + 1 < len(raw):
            argv.append(f"--ibtracs-normalize-lon={raw[i+1].strip()}")
            skip = True
        elif tok.startswith("--ibtracs-normalize-lon="):
            lhs, rhs = tok.split("=", 1)
            argv.append(f"{lhs}={rhs.strip()}")
        else:
            argv.append(tok)
    return argv


def parse_args():
    ap = argparse.ArgumentParser(description="Alias for reporting_v2 summary.")
    ap.add_argument("--run-name", required=True, help="Logical name/label for this run.")
    ap.add_argument("--out-dir", default="results/reports", help="Per-run output folder.")
    ap.add_argument("--seed-summary", default="results/seedmaps/seed_summary.txt")
    ap.add_argument("--seed-analysis", default=None)
    ap.add_argument("--alerts-dir", default="results/alerts")
    ap.add_argument(
        "--conversion-csv",
        "--include-conversion",
        dest="conversion_csv",
        default=None,
    )
    ap.add_argument("--viability-targets", default="data/grid_train_gse_panel_targets.parquet")
    ap.add_argument("--viability-metrics", default="models/viability_model_metrics.json")
    ap.add_argument("--viability-thresholds", default="results/sweeps/viability_best_thresholds.csv")
    ap.add_argument("--storm-timeseries", default=None)
    ap.add_argument("--seed-union", default=None)
    ap.add_argument("--viability-horizons", default="24,48,72,120")
    ap.add_argument("--ibtracs", default=None)
    ap.add_argument("--ibtracs-area", default=None)
    ap.add_argument(
        "--ibtracs-normalize-lon",
        choices=["none", "-180..180", "0..360"],
        default="-180..180",
    )
    ap.add_argument("--extras", default="{}")
    ap.add_argument("--chunk-rows", type=int, default=None)
    ap.add_argument("--chunksize", type=int, default=None)
    ap.add_argument("--parquet-rows", type=int, default=None)
    return ap.parse_args(_normalize_argv(sys.argv[1:]))


def _maybe_add(cmd: list[str], flag: str, value: str | None) -> None:
    if value is None or value == "":
        return
    cmd.extend([flag, str(value)])


def main() -> int:
    args = parse_args()
    script = HERE / "reporting_v2.py"
    cmd = [
        sys.executable,
        str(script),
        "--run-name",
        args.run_name,
        "--out-dir",
        str(args.out_dir),
        "--write-txt",
        "--txt-only",
    ]

    _maybe_add(cmd, "--seed-summary", args.seed_summary)
    _maybe_add(cmd, "--seed-analysis", args.seed_analysis)
    _maybe_add(cmd, "--alerts-dir", args.alerts_dir)
    _maybe_add(cmd, "--conversion-csv", args.conversion_csv)
    _maybe_add(cmd, "--viability-targets", args.viability_targets)
    _maybe_add(cmd, "--viability-metrics", args.viability_metrics)
    _maybe_add(cmd, "--viability-thresholds", args.viability_thresholds)
    _maybe_add(cmd, "--storm-timeseries", args.storm_timeseries)
    _maybe_add(cmd, "--seed-union", args.seed_union)
    _maybe_add(cmd, "--viability-horizons", args.viability_horizons)
    _maybe_add(cmd, "--ibtracs", args.ibtracs)
    _maybe_add(cmd, "--ibtracs-area", args.ibtracs_area)
    if args.ibtracs_normalize_lon:
        cmd.append(f"--ibtracs-normalize-lon={args.ibtracs_normalize_lon}")
    _maybe_add(cmd, "--extras", args.extras)

    print("[summary] report_generate_summary is deprecated; delegating to reporting_v2.py")
    proc = subprocess.run(cmd)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

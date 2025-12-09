#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viability_pipeline.py

Run the three core alert steps for the viability model:
  1) apply_thresholds.py   -> base alerts (prob_viable + alert_base)
  2) throttle_by_percentile.py -> alert_throttled
  3) denoise_alerts.py     -> alert_final

Defaults align with the new viability/t_to_storm flow and file layout:
  labelled : data/grid_labelled_FMA_gka_realthermo_sph_ms_id.parquet
  model    : models/viability_model.pkl
  outputs  : results/alerts/alerts_<run>_{base,thr,final}.parquet
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _strip_choice(val: str) -> str:
    return str(val).strip()


def run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd).returncode


def parse_args():
    ap = argparse.ArgumentParser(
        description="End-to-end viability alerts pipeline (apply threshold -> throttle -> denoise).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--run-name", required=True, help="Run name for stamping outputs (alerts_<run>_*).")
    ap.add_argument(
        "--labelled",
        default="data/grid_labelled_FMA_gka_realthermo_sph_ms_id.parquet",
        help="Labelled/scoring table with required features.",
    )
    ap.add_argument("--model", default="models/viability_model.pkl", help="Trained viability model bundle.")
    ap.add_argument("--thr", type=float, default=0.15, help="Probability threshold for base alerts.")
    ap.add_argument("--keep-quantile", type=float, default=0.90, help="Per-hour fraction to keep after throttle.")
    ap.add_argument("--persist-hours", type=int, default=3, help="Temporal persistence hours in denoise.")
    ap.add_argument("--min-neighbors", type=int, default=3, help="Spatial neighbor requirement in denoise.")
    ap.add_argument("--connectivity", type=int, choices=[4, 8], default=4, help="Spatial connectivity for denoise.")
    ap.add_argument("--normalize-lon", default="-180..180", choices=["none", "-180..180", "0..360"], type=_strip_choice)
    ap.add_argument("--prob-col", default="prob_viable", help="Probability column name.")
    ap.add_argument("--flag-col-base", default="alert_base", help="Flag name after thresholding.")
    ap.add_argument("--flag-col-thr", default=None, help="Flag name after throttle (defaults to flag-col-base).")
    ap.add_argument("--base-out", default=None, help="Override base output path.")
    ap.add_argument("--thr-out", default=None, help="Override throttled output path.")
    ap.add_argument("--out", default=None, help="Override final output path.")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    # Compatibility: accept chunking hints without using them
    ap.add_argument("--chunk-rows", type=int, default=None, help="Ignored; accepted for pipeline compatibility.")
    ap.add_argument("--chunksize", type=int, default=None, help="Alias for --chunk-rows (ignored).")
    ap.add_argument("--parquet-rows", type=int, default=None, help="Ignored; accepted for pipeline compatibility.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    base_out = args.base_out or f"results/alerts/alerts_{args.run_name}_base.parquet"
    thr_out = args.thr_out or f"results/alerts/alerts_{args.run_name}_thr.parquet"
    final_out = args.out or f"results/alerts/alerts_{args.run_name}_final.parquet"
    flag_thr = args.flag_col_thr or args.flag_col_base

    steps = [
        [
            sys.executable,
            str(Path(__file__).with_name("apply_thresholds.py")),
            "--labelled",
            args.labelled,
            "--model",
            args.model,
            "--thr",
            str(args.thr),
            "--prob-col",
            args.prob_col,
            "--flag-col",
            args.flag_col_base,
            "--normalize-lon",
            args.normalize_lon,
            "--out",
            base_out,
        ],
        [
            sys.executable,
            str(Path(__file__).with_name("throttle_by_percentile.py")),
            "--alerts",
            base_out,
            "--out",
            thr_out,
            "--prob-col",
            args.prob_col,
            "--score-col",
            args.prob_col,
            "--flag-col",
            flag_thr,
            "--keep-quantile",
            str(args.keep_quantile),
            "--normalize-lon",
            args.normalize_lon,
        ],
        [
            sys.executable,
            str(Path(__file__).with_name("denoise_alerts.py")),
            "--alerts",
            thr_out,
            "--out",
            final_out,
            "--flag-col",
            flag_thr,
            "--flag-out",
            "alert_final",
            "--score-col",
            args.prob_col,
            "--persist-hours",
            str(args.persist_hours),
            "--min-neighbors",
            str(args.min_neighbors),
            "--connectivity",
            str(args.connectivity),
            "--overwrite",
        ],
    ]

    for cmd in steps:
        print(f"\n[viability-pipeline] step -> {' '.join(cmd)}")
        if args.dry_run:
            continue
        rc = run(cmd)
        if rc != 0:
            print(f"[viability-pipeline] step failed with code {rc}", flush=True)
            return rc

    if args.dry_run:
        print("[viability-pipeline] dry run complete (no commands executed).", flush=True)
        return 0

    print(f"[viability-pipeline] done. final -> {final_out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reports_and_maps_manager.py

One-stop orchestrator to bundle maps + summary + sanity checks
into a per-run folder like:

  results/reports/251122_run001/

Folders increment (run001, run002, …) per day so you can stack runs
for comparisons without overwriting prior outputs.

It wires together:
  - report_make_maps.py
  - plot_seed_map_cartopy.py
  - plot_seeds_with_ibtracs.py
  - plot_seed_track_map_cartopy.py
  - report_generate_summary.py
  - report_sanity_checks.py

Typical usage
-------------
python reports_and_maps_manager.py \
  --run-name coral_sea_demo_lead72 \
  --union-csv   results/seedmaps/coral_sea_demo_union_byhour.csv \
  --patches-csv results/seedmaps/seed_patches.csv \
  --matches-csv results/seedmaps/seed_track_matches.csv \
  --seed-summary  results/seedmaps/seed_summary.txt \
  --seed-analysis results/seedmaps/seed_analysis.txt \
  --alerts-dir    results/alerts \
  --conversion-csv results/seedmaps/coral_sea_demo_conversion_rates.csv \
  --ibtracs data/tracks/ibtracs.ALL.list.v04r01.csv
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


HERE = Path(__file__).resolve().parent


# --------------------- run-folder helpers ---------------------


def _normalize_date_str(date_str: str | None) -> str:
    """Return YYMMDD date string.

    Accepts explicit YYMMDD or YYYYMMDD (trimmed to YYMMDD) to keep
    backwards compatibility with older docs/CLI habits.
    """
    if date_str is None:
        return datetime.utcnow().strftime("%y%m%d")

    clean = str(date_str).strip()
    if clean.isdigit() and len(clean) == 8:
        clean = clean[2:]

    if not (clean.isdigit() and len(clean) == 6):
        raise ValueError("run-date must be YYMMDD or YYYYMMDD (digits only)")

    return clean


def make_run_dir(root: Path, date_str: str | None) -> Path:
    """
    Create a new run directory under `root` with pattern:
      YYMMDD_runNNN

    Returns the newly created directory.
    """
    date_str = _normalize_date_str(date_str)

    root.mkdir(parents=True, exist_ok=True)

    prefix = f"{date_str}_run"
    existing_nums: List[int] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix) :]
        try:
            n = int(suffix)
        except ValueError:
            continue
        existing_nums.append(n)

    next_n = max(existing_nums) + 1 if existing_nums else 1
    run_dir = root / f"{prefix}{next_n:03d}"
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def build_cmd(script: Path, args: List[str]) -> List[str]:
    return [sys.executable, str(script), *args]


def run_step(tag: str, script: Path, args: List[str]) -> Tuple[bool, int]:
    """
    Run a single subprocess step; return (ok, returncode).
    """
    cmd = build_cmd(script, args)
    print(f"\n[manager] STEP {tag}:")
    print("  $ " + " ".join(str(x) for x in cmd))
    res = subprocess.run(cmd)
    ok = (res.returncode == 0)
    if not ok:
        print(f"[manager] STEP {tag} FAILED with code {res.returncode}")
    return ok, res.returncode


# --------------------- main orchestration ---------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Bundle per-run maps + summary + sanity checks into a dated run folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core run identity
    ap.add_argument(
        "--run-name",
        required=True,
        help="Human-readable name/tag for this run (used in summary filename).",
    )
    ap.add_argument(
        "--run-root",
        default="results/reports",
        help="Root folder under which per-run folders YYMMDD_runNNN are created.",
    )
    ap.add_argument(
        "--run-date",
        default=None,
        help="Date prefix for run folder (YYMMDD or YYYYMMDD). Default: today (UTC).",
    )

    # Key inputs shared across reports
    ap.add_argument(
        "--union-csv",
        default="results/seedmaps/seed_union_byhour.csv",
        help="Seed union-by-hour CSV for maps.",
    )
    ap.add_argument(
        "--patches-csv",
        default="results/seedmaps/seed_patches.csv",
        help="Seed patches CSV (centroids) for quick QA maps.",
    )
    ap.add_argument(
        "--matches-csv",
        default="results/seedmaps/seed_track_matches.csv",
        help="Seed–track matches CSV for overlays.",
    )
    ap.add_argument(
        "--seed-summary",
        default="results/seedmaps/seed_summary.txt",
        help="Seed-level summary text produced earlier in the pipeline.",
    )
    ap.add_argument(
        "--seed-analysis",
        default=None,
        help="Optional seed analysis text report (e.g. from analyze_seeds.py).",
    )
    ap.add_argument(
        "--alerts-dir",
        default="results/alerts",
        help="Alerts directory for snapshot + sanity checks.",
    )
    ap.add_argument(
        "--conversion-csv",
        default="results/seedmaps/coral_sea_demo_conversion_rates.csv",
        help="Optional conversion/proto-outcomes CSV to embed in summary.",
    )

    # IBTrACS / storm overlays
    ap.add_argument(
        "--ibtracs",
        default=None,
        help="IBTrACS CSV for overlays (if omitted, IBTrACS maps are skipped).",
    )
    ap.add_argument(
        "--ibtracs-area",
        default=None,
        help='Optional AOI for IBTrACS overlays: "latN,lonW,latS,lonE".',
    )
    ap.add_argument(
        "--ibtracs-normalize-lon",
        choices=["none", "-180..180", "0..360"],
        default="-180..180",
        help="Longitude frame to use for IBTrACS overlays.",
    )

    # Behaviour toggles
    ap.add_argument(
        "--skip-quick-maps",
        action="store_true",
        help="Skip report_make_maps quick QA maps.",
    )
    ap.add_argument(
        "--skip-cartopy-seed-map",
        action="store_true",
        help="Skip plot_seed_map_cartopy (cartopy seed map).",
    )
    ap.add_argument(
        "--skip-ibtracs-maps",
        action="store_true",
        help="Skip IBTrACS-related plots (even if --ibtracs is given).",
    )
    ap.add_argument(
        "--skip-sanity",
        action="store_true",
        help="Skip final sanity checks step.",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="If set, stop on first failing step and exit with its code.",
    )

    args = ap.parse_args()

    run_root = Path(args.run_root)
    run_dir = make_run_dir(run_root, args.run_date)
    maps_dir = run_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    print(f"[manager] Run folder: {run_dir}")

    # --- STEP 1: quick QA maps (simple scatter maps) ---
    if not args.skip_quick_maps:
        script = HERE / "report_make_maps.py"
        step_args = [
            "--union-csv", args.union_csv,
            "--patches-csv", args.patches_csv,
            "--out-dir", str(maps_dir / "quick"),
        ]
        ok, code = run_step("quick-maps", script, step_args)
        if not ok and args.strict:
            return code

    # --- STEP 2: cartopy seed map from union CSV ---
    if not args.skip_cartopy_seed_map:
        script = HERE / "plot_seed_map_cartopy.py"
        out_png = maps_dir / "seeds_union_cartopy.png"
        step_args = [
            "--seeds", args.union_csv,
            "--out-png", str(out_png),
            "--title", f"Seeds (union by hour) — {args.run_name}",
        ]
        ok, code = run_step("seed-map-cartopy", script, step_args)
        if not ok and args.strict:
            return code

    # --- STEP 3: IBTrACS + seeds overlay ---
    if (args.ibtracs is not None) and (not args.skip_ibtracs_maps):
        script = HERE / "plot_seeds_with_ibtracs.py"
        out_png = maps_dir / "seeds_with_ibtracs.png"
        step_args = [
            "--ibtracs", args.ibtracs,
            "--out-png", str(out_png),
        ]
        # prefer matches if present; otherwise union seeds
        matches_path = Path(args.matches_csv)
        if matches_path.exists():
            step_args += ["--matches", str(matches_path)]
        else:
            step_args += ["--seeds", args.union_csv]

        if args.ibtracs_area:
            step_args += ["--area", args.ibtracs_area]
        if args.ibtracs_normalize_lon:
            step_args += ["--normalize-lon", args.ibtracs_normalize_lon]

        ok, code = run_step("seeds-with-ibtracs", script, step_args)
        if not ok and args.strict:
            return code

    # --- STEP 4: seed–track match map (cartopy) ---
    if (not args.skip_ibtracs_maps) and Path(args.matches_csv).exists():
        script = HERE / "plot_seed_track_map_cartopy.py"
        out_png = maps_dir / "seed_track_map.png"
        step_args = [
            "--matches", args.matches_csv,
            "--out", str(out_png),
        ]
        ok, code = run_step("seed-track-map", script, step_args)
        if not ok and args.strict:
            return code

    # --- STEP 5: summary text report ---
    script = HERE / "report_generate_summary.py"
    step_args = [
        "--run-name", args.run_name,
        "--out-dir", str(run_dir),
        "--seed-summary", args.seed_summary,
        "--alerts-dir", args.alerts_dir,
        "--include-conversion", args.conversion_csv,
    ]
    if args.seed_analysis:
        step_args += ["--seed-analysis", args.seed_analysis]

    ok, code = run_step("summary", script, step_args)
    if not ok and args.strict:
        return code

    # --- STEP 6: sanity checks over key artifacts ---
    if not args.skip_sanity:
        script = HERE / "report_sanity_checks.py"

        # Files we *expect* after a healthy run
        # - core seed artifacts (original run)
        # - per-run summary
        must_exist = [
            args.seed_summary,
            args.patches_csv,
            args.matches_csv,
        ]
        # summary text lives in run_dir, file name driven by run-name logic
        safe_run = args.run_name.strip().replace(" ", "_")
        summary_txt = run_dir / f"{safe_run}_report.txt"
        must_exist.append(str(summary_txt))

        step_args = [
            "--must-exist",
            *must_exist,
            "--alerts-dir", args.alerts_dir,
            "--require-alerts",
            "--strict",  # internal strict w.r.t its own checks
        ]
        ok, code = run_step("sanity-checks", script, step_args)
        if not ok and args.strict:
            return code

    print(f"\n[manager] Completed. Run artifacts in: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
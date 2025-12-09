#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reports_and_maps_manager.py

One-stop orchestrator to bundle maps + summary + sanity checks
into a per-run folder like:

  results/reports/20251122_run001/

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
  --run-name coral_sea_demo \
  --union-csv   results/seedmaps/coral_sea_demo_union_byhour.csv \
  --patches-csv results/seedmaps/coral_sea_demo_seed_patches.csv \
  --matches-csv results/seedmaps/coral_sea_demo_seed_track_matches.csv \
  --seed-summary  results/seedmaps/coral_sea_demo_seed_summary.txt \
  --seed-analysis results/seedmaps/coral_sea_demo_seed_analysis.txt \
  --alerts-dir    results/alerts \
  --conversion-csv results/seedmaps/coral_sea_demo_conversion_rates.csv \
  --viability-thresholds results/sweeps/viability_best_thresholds.csv \
  --ibtracs data/tracks/tracks_subset.csv
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


def make_run_dir(root: Path, date_str: str | None) -> Path:
    """
    Create a new run directory under `root` with pattern:
      YYYYMMDD_runNNN

    Returns the newly created directory.
    """
    if date_str is None:
        date_str = datetime.utcnow().strftime("%Y%m%d")

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


def _parse_leads(spec: str) -> List[int]:
    parts = [p.strip() for p in spec.split(",") if p.strip()] if spec else []
    leads = []
    for p in parts:
        try:
            leads.append(int(p))
        except Exception:
            continue
    return leads


# --------------------- main orchestration ---------------------


def main() -> int:
    # Support mode-first calls (e.g., "summary") as used by run_pipeline.
    argv = sys.argv[1:]
    mode_first = None
    if argv and not argv[0].startswith("--"):
        mode_first = argv[0]
        argv = argv[1:]

    # Convenience dispatch for summary-only calls so pipeline can do:
    #   reports_and_maps_manager.py summary --out-md ...
    if mode_first == "summary":
        ignore_flags = {"--chunk-rows", "--chunksize", "--parquet-rows"}
        args_iter = iter(argv)
        passthrough: List[str] = []
        run_name = "auto"
        out_dir = Path("results/reports")
        out_md = None
        for tok in args_iter:
            if tok in ignore_flags:
                # skip value if present
                nxt = next(args_iter, None)
                continue
            if tok == "--run-name":
                run_name = next(args_iter, run_name)
                continue
            if tok == "--out-dir":
                out_dir = Path(next(args_iter, str(out_dir)))
                continue
            if tok == "--out-md":
                out_md = Path(next(args_iter, "results/reports/summary_auto.md"))
                continue
            passthrough.append(tok)
        if "--run-name" not in passthrough:
            passthrough = ["--run-name", run_name] + passthrough
        if "--out-dir" not in passthrough:
            passthrough = ["--out-dir", str(out_dir)] + passthrough
        script = HERE / "report_generate_summary.py"
        ok, code = run_step("summary", script, passthrough)
        if ok and out_md:
            safe = Path(out_dir) / f"{run_name.strip().replace(' ', '_')}_report.txt"
            try:
                out_md.parent.mkdir(parents=True, exist_ok=True)
                out_md.write_text(safe.read_text())
                print(f"[manager] copied summary to {out_md}")
            except Exception as e:
                print(f"[manager] warning: could not copy summary to {out_md}: {e}")
        return code

    ap = argparse.ArgumentParser(
        description="Bundle per-run maps + summary + sanity checks into a dated run folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core run identity
    ap.add_argument(
        "--run-name",
        default="auto",
        help="Human-readable name/tag for this run (used in summary filename).",
    )
    ap.add_argument(
        "--run-root",
        default="results/reports",
        help="Root folder under which per-run folders YYYYMMDD_runNNN are created.",
    )
    ap.add_argument(
        "--run-date",
        default=None,
        help="Date prefix for run folder (YYYYMMDD). Default: today (UTC).",
    )

    # Key inputs shared across reports
    ap.add_argument(
    "--union-csv",
        default=None,
        help="Seed union-by-hour CSV for maps (default: results/seedmaps/<run>_union_byhour.csv).",
    )
    ap.add_argument(
        "--patches-csv",
        default=None,
        help="Seed patches CSV (centroids) for quick QA maps (default: results/seedmaps/<run>_seed_patches.csv).",
    )
    ap.add_argument(
        "--matches-csv",
        default=None,
        help="Seed-track matches CSV for overlays (default: results/seedmaps/<run>_seed_track_matches.csv).",
    )
    ap.add_argument(
        "--seed-summary",
        default=None,
        help="Seed-level summary text produced earlier in the pipeline (default: results/seedmaps/<run>_seed_summary.txt).",
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
        default=None,
        help="Optional conversion/proto-outcomes CSV to embed in summary (default: results/seedmaps/<run>_conversion_rates.csv if present).",
    )
    ap.add_argument(
        "--viability-thresholds",
        default=None,
        help="Optional viability thresholds CSV (default: results/sweeps/viability_best_thresholds.csv).",
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

    # Slow-tick diagnostics (optional)
    ap.add_argument(
        "--run-slowtick",
        action="store_true",
        help="Run slowtick_diagnostics.py on alerts (optional diagnostic stage).",
    )
    ap.add_argument(
        "--slowtick-alerts-dir",
        default="results/alerts_throttled",
        help="Alerts directory for slow-tick diagnostics.",
    )
    ap.add_argument(
        "--slowtick-run-name",
        default=None,
        help="Run name used in alerts filenames (defaults to --run-name).",
    )
    ap.add_argument(
        "--slowtick-leads",
        default="24,48,72,120,240",
        help="Comma-separated lead hours to include in diagnostics.",
    )
    ap.add_argument(
        "--slowtick-flag-col",
        default="alert_final",
        help="Flag column to use (tries fallbacks if missing).",
    )
    ap.add_argument(
        "--slowtick-out-subdir",
        default="slowtick",
        help="Subdirectory under the run folder for slow-tick outputs.",
    )
    ap.add_argument(
        "--slowtick-normalize-lon",
        choices=["none", "-180..180", "0..360"],
        default="-180..180",
        help="Longitude normalization for diagnostics input.",
    )
    ap.add_argument(
        "--slowtick-area",
        default=None,
        help='Optional AOI "latN,lonW,latS,lonE" for diagnostics.',
    )
    ap.add_argument(
        "--slowtick-time-format",
        default=None,
        help="Optional strptime for parsing alert times.",
    )
    ap.add_argument(
        "--slowtick-prefer",
        choices=["throttled", "denoised", "base"],
        default="throttled",
        help="Preferred alerts stage if multiple files exist.",
    )
    ap.add_argument(
        "--slowtick-save-timeseries",
        action="store_true",
        help="Save hourly global coverage time series.",
    )
    ap.add_argument(
        "--slowtick-save-hemi-timeseries",
        action="store_true",
        help="Save hourly N/S coverage time series.",
    )
    ap.add_argument(
        "--slowtick-bootstrap-B",
        type=int,
        default=500,
        help="Bootstrap reps for CIs (knee/parity).",
    )
    ap.add_argument(
        "--slowtick-min-hours-per-lead",
        type=int,
        default=8,
        help="Minimum hourly points required per lead.",
    )
    ap.add_argument(
        "--slowtick-fft-gap-fill",
        type=int,
        default=2,
        help="Fill NaN gaps up to this length before FFT (hours).",
    )
    ap.add_argument(
        "--slowtick-debug",
        action="store_true",
        help="Verbose file/range debug for diagnostics.",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="If set, stop on first failing step and exit with its code.",
    )
    # Pipeline compatibility: accept chunking hints even though manager doesn't use them.
    ap.add_argument("--chunk-rows", type=int, default=None, help="Ignored; accepted for pipeline compatibility.")
    ap.add_argument("--chunksize", type=int, default=None, help="Alias for --chunk-rows (ignored).")
    ap.add_argument("--parquet-rows", type=int, default=None, help="Ignored; accepted for pipeline compatibility.")

    args = ap.parse_args(argv)

    run_root = Path(args.run_root)
    run_dir = make_run_dir(run_root, args.run_date)
    maps_dir = run_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    print(f"[manager] Run folder: {run_dir}")

    def or_default(path_arg: str | None, pattern: str | None, fallback: str | None = None) -> str:
        if path_arg:
            return path_arg
        if pattern and args.run_name:
            return pattern.format(run=args.run_name)
        return fallback or ""

    union_csv = or_default(args.union_csv, "results/seedmaps/{run}_union_byhour.csv")
    patches_csv = or_default(args.patches_csv, "results/seedmaps/{run}_seed_patches.csv")
    matches_csv = or_default(args.matches_csv, "results/seedmaps/{run}_seed_track_matches.csv")
    seed_summary = or_default(args.seed_summary, "results/seedmaps/{run}_seed_summary.txt")
    seed_analysis = or_default(args.seed_analysis, "results/seedmaps/{run}_seed_analysis.txt", "")
    conversion_csv = or_default(args.conversion_csv, "results/seedmaps/{run}_conversion_rates.csv", "")
    viability_thr = args.viability_thresholds or "results/sweeps/viability_best_thresholds.csv"
    ibtracs_default = "data/tracks/tracks_subset.csv"
    ibtracs_path = args.ibtracs or (ibtracs_default if Path(ibtracs_default).exists() else None)

    # --- STEP 1: quick QA maps (simple scatter maps) ---
    if not args.skip_quick_maps:
        script = HERE / "report_make_maps.py"
        step_args = [
            "--union-csv", union_csv,
            "--patches-csv", patches_csv,
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
            "--seeds", union_csv,
            "--out-png", str(out_png),
            "--title", f"Seeds (union by hour) — {args.run_name}",
        ]
        ok, code = run_step("seed-map-cartopy", script, step_args)
        if not ok and args.strict:
            return code

    # --- STEP 3: IBTrACS + seeds overlay ---
    if (ibtracs_path is not None) and (not args.skip_ibtracs_maps):
        script = HERE / "plot_seeds_with_ibtracs.py"
        out_png = maps_dir / "seeds_with_ibtracs.png"
        step_args = [
            "--ibtracs", ibtracs_path,
            "--out-png", str(out_png),
        ]
        # prefer matches if present; otherwise union seeds
        matches_path = Path(matches_csv)
        if matches_path.exists():
            step_args += ["--matches", str(matches_path)]
        else:
            step_args += ["--seeds", union_csv]

        if args.ibtracs_area:
            step_args += ["--area", args.ibtracs_area]
        if args.ibtracs_normalize_lon:
            step_args += ["--normalize-lon", args.ibtracs_normalize_lon]

        ok, code = run_step("seeds-with-ibtracs", script, step_args)
        if not ok and args.strict:
            return code

    # --- STEP 4: seed–track match map (cartopy) ---
    if (not args.skip_ibtracs_maps) and Path(matches_csv).exists():
        script = HERE / "plot_seed_track_map_cartopy.py"
        out_png = maps_dir / "seed_track_map.png"
        step_args = [
            "--matches", matches_csv,
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
        "--seed-summary", seed_summary,
        "--alerts-dir", args.alerts_dir,
        "--viability-thresholds", viability_thr,
    ]
    if seed_analysis:
        step_args += ["--seed-analysis", seed_analysis]
    if conversion_csv:
        step_args += ["--include-conversion", conversion_csv]
    if ibtracs_path:
        step_args += ["--ibtracs", ibtracs_path]
    if args.ibtracs_area:
        step_args += ["--ibtracs-area", args.ibtracs_area]
    if args.ibtracs_normalize_lon:
        step_args += ["--ibtracs-normalize-lon", args.ibtracs_normalize_lon]

    ok, code = run_step("summary", script, step_args)
    if not ok and args.strict:
        return code

    # --- Optional: slow-tick diagnostics on alerts ---
    if args.run_slowtick:
        leads = _parse_leads(args.slowtick_leads)
        if not leads:
            print("[manager] slowtick: no valid leads parsed; skipping.")
        else:
            script = HERE / "slowtick_diagnostics.py"
            out_dir = Path(args.slowtick_out_subdir)
            if not out_dir.is_absolute():
                out_dir = run_dir / out_dir
            step_args = [
                "--alerts-dir", args.slowtick_alerts_dir,
                "--run-name", args.slowtick_run_name or args.run_name,
                "--leads", *map(str, leads),
                "--flag-col", args.slowtick_flag_col,
                "--out-dir", str(out_dir),
                "--normalize-lon", args.slowtick_normalize_lon,
                "--prefer", args.slowtick_prefer,
                "--bootstrap-B", str(args.slowtick_bootstrap_B),
                "--min-hours-per-lead", str(args.slowtick_min_hours_per_lead),
                "--fft-gap-fill", str(args.slowtick_fft_gap_fill),
            ]
            if args.slowtick_area:
                step_args += ["--area", args.slowtick_area]
            if args.slowtick_time_format:
                step_args += ["--time-format", args.slowtick_time_format]
            if args.slowtick_save_timeseries:
                step_args.append("--save-timeseries")
            if args.slowtick_save_hemi_timeseries:
                step_args.append("--save-hemi-timeseries")
            if args.slowtick_debug:
                step_args.append("--debug")
            ok, code = run_step("slowtick-diag", script, step_args)
            if not ok and args.strict:
                return code

    # --- STEP 6: sanity checks over key artifacts ---
    if not args.skip_sanity:
        script = HERE / "report_sanity_checks.py"

        # Files we *expect* after a healthy run
        # - core seed artifacts (original run)
        # - per-run summary
        must_exist = [
            seed_summary,
            patches_csv,
            matches_csv,
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

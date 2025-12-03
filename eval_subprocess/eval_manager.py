#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_manager.py

Unified front door for evaluation & core plotting tools:

  - eval_alert_hits.py            (hits)
  - eval_leadtime_grid.py         (leadtime)
  - eval_leadtime_grid_progress.py(leadtime-progress)
  - eval_leadtime_motion.py       (motion)
  - hourly_metrics.py             (hourly-metrics)
  - hourly_rollup.py              (hourly-rollup)
  - plot_skill_vs_tracks.py       (plot-skill)
  - per_storm_overlays.py         (per-storm)

Usage examples
--------------

# 1) Evaluate alert hits vs truth (lead-aware)
python eval_manager.py hits \
  --labelled data/labelled_grid.parquet \
  --alerts   results/alerts/alerts_run_lead72.csv.gz \
  --lead-hours 72 \
  --flag-col alert_final \
  --truth-col storm_window \
  --normalize-lon -180..180 \
  --out-csv results/eval/alerts_run_lead72_eval.csv

# 2) Lead-time skill on the grid (strict future labels)
python eval_manager.py leadtime \
  --labelled data/labelled_grid.parquet \
  --model    models/gka_bundle.joblib \
  --target   storm \
  --lead-hours 6 12 24 48

# 3) Lead-time skill with checkpoints / per-lead models
python eval_manager.py leadtime-progress \
  --labelled data/labelled_grid.parquet \
  --model    models/gka_bundle.joblib \
  --target   storm \
  --lead-hours 24 48 72 \
  --checkpoint-dir checkpoints/lead_eval

# 4) Motion-aware lead-time evaluation (spatially dilated labels)
python eval_manager.py motion \
  --labelled data/labelled_grid.parquet \
  --model    models/gka_motion.joblib \
  --target   storm \
  --lead-hours 24 48 \
  --neighbor-radius 1 \
  --advect-cells 1 \
  --wrap-lon \
  --checkpoint-dir checkpoints/motion_eval

# 5) Hourly alert coverage & cluster stats
python eval_manager.py hourly-metrics \
  --alerts results/alerts/alerts_run_lead72_thr*.csv.gz \
  --flag-col alert_final \
  --prob-col prob \
  --normalize-lon -180..180 \
  --tag lead72_thr \
  --out results/eval/hourly_metrics_lead72.csv

# 6) Hourly rollup of risks/flags
python eval_manager.py hourly-rollup \
  --alerts results/alerts/alerts_run_lead72_thr0.04.csv.gz \
  --flag-col alert_final \
  --risk-cols risk_final,risk_geom \
  --out results/eval/hourly_rollup_lead72.parquet

# 7) Plot skill vs tracks (per-hour coverage/F1 + IBTrACS presence)
python eval_manager.py plot-skill \
  --lead-hours 72 \
  --metrics-file results/per_hour/metrics_lead72.parquet \
  --tracks data/tracks/ibtracs_subset.csv \
  --normalize-lon -180..180 \
  --area "-5,140,-30,170" \
  --out-dir results/per_hour/plots \
  --smooth-k 5

# 8) Per-storm overlays (IBTrACS + seed starts)
python eval_manager.py per-storm \
  --matches results/seedmaps/seed_track_matches.csv \
  --ibtracs data/tracks/ibtracs.ALL.list.v04r01.csv \
  --normalize-lon -180..180 \
  --area "-5,140,-30,170" \
  --out-dir results/per_storm \
  --overlay-prob prob_max \
  --save-per-storm-csv
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict

HERE = Path(__file__).resolve().parent

# Map high-level tools -> script filenames in this directory
SCRIPT_MAP: Dict[str, str] = {
    # Core evals
    "hits":              "eval_alert_hits.py",
    "leadtime":          "eval_leadtime_grid.py",
    "leadtime-progress": "eval_leadtime_grid_progress.py",
    "motion":            "eval_leadtime_motion.py",

    # Hourly summaries
    "hourly-metrics":    "hourly_metrics.py",
    "hourly-rollup":     "hourly_rollup.py",

    # Plots / overlays
    "plot-skill":        "plot_skill_vs_tracks.py",
    "per-storm":         "per_storm_overlays.py",
}

# Friendly aliases and shorthands
ALIASES: Dict[str, str] = {
    "leadtime-grid": "leadtime",
    "lt":            "leadtime",
    "lt-progress":   "leadtime-progress",
    "progress":      "leadtime-progress",
    "hm":            "hourly-metrics",
    "hr":            "hourly-rollup",
    "skill":         "plot-skill",
    "overlays":      "per-storm",
    # (no more seed/seed-map aliases here; those live under reports_subprocess now)
}

def normalize_tool(name: str) -> str:
    """Resolve aliases and minor dash/underscore drift."""
    name = name.strip()
    if name in SCRIPT_MAP:
        return name
    if name in ALIASES:
        return ALIASES[name]
    # allow dashed/underscored drift
    alt = name.replace("_", "-")
    if alt in SCRIPT_MAP:
        return alt
    alt2 = name.replace("-", "_")
    if alt2 in SCRIPT_MAP:
        return alt2
    return name  # fall back (will raise in find_script if unknown)

def find_script(tool: str) -> Path:
    tool = normalize_tool(tool)
    if tool in SCRIPT_MAP:
        p = (HERE / SCRIPT_MAP[tool]).resolve()
        if p.exists():
            return p

    # Forgiving fallbacks: try raw tool name variants
    candidates = [
        tool + ".py",
        tool.replace("-", "_") + ".py",
        tool.replace("_", "-") + ".py",
    ]
    for fn in candidates:
        p = (HERE / fn).resolve()
        if p.exists():
            return p

    tried = [SCRIPT_MAP.get(tool, "<no direct map>")] + candidates
    raise FileNotFoundError(
        f"Could not locate a script for subcommand '{tool}' in {HERE}\n"
        f"Tried: {tried}"
    )

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Evaluation & plotting manager (hits, lead-time, hourly metrics, skill vs tracks, per-storm overlays)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "tool",
        choices=sorted(set(list(SCRIPT_MAP.keys()) + list(ALIASES.keys()))),
        help="Which evaluation/plot tool to run",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command and exit without running it.",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Append --quiet to the downstream tool if it supports it.",
    )

    # Parse known to allow pass-through of everything else
    ns, extra = ap.parse_known_args()

    script = find_script(ns.tool)
    cmd = [sys.executable, str(script)]
    if ns.quiet:
        cmd.append("--quiet")
    cmd.extend(extra)

    print(f"\n$ {' '.join(map(str, cmd))}")
    if ns.dry_run:
        return 0

    proc = subprocess.run(cmd)
    return proc.returncode

if __name__ == "__main__":
    sys.exit(main())
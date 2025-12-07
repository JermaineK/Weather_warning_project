#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report_generate_summary.py

Generate a consolidated plain-text summary for a single run.

Intended usage:
  - A higher-level manager (e.g. reports_and_maps_manager.py) creates a
    unique per-run report folder like:
        results/reports/20251122_run001
  - That manager then calls this script with:
        --run-name 20251122_run001
        --out-dir  results/reports/20251122_run001
        --seed-summary <path>
        --seed-analysis <path>
        --viability-targets <path>
        --viability-metrics <path>
        --extras '{"key": "path", ...}'

This script assumes:
  - "run-name" is just a logical label shown in the header.
  - "out-dir" is already unique for this run; we just write a report
    file inside it and do not try to deduplicate names.
"""

import argparse
import json
from pathlib import Path

import numpy as np  # noqa: F401 (kept in case we expand stats later)
import pandas as pd


def read_text_safe(p: Path) -> str:
    """Read a text file as UTF-8; return '' on any failure."""
    try:
        if not p or not p.exists() or p.stat().st_size == 0:
            return ""
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


def exists_nonempty(p: Path) -> bool:
    return bool(p and p.exists() and p.stat().st_size > 0)


def slugify_name(name: str) -> str:
    """
    Turn a run name into a safe-ish filename chunk.
    Keep it simple: alnum, dash, underscore; collapse others to '_'.
    """
    safe_chars = []
    for ch in str(name):
        if ch.isalnum() or ch in "-_":
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    slug = "".join(safe_chars).strip("_")
    return slug or "run"


def main():
    ap = argparse.ArgumentParser(description="Generate consolidated run summary.")
    ap.add_argument("--run-name", required=True, help="Logical name/label for this run.")
    ap.add_argument(
        "--out-dir",
        default="results/reports",
        help="Per-run output folder (manager should make this unique per run).",
    )
    ap.add_argument(
        "--seed-summary",
        default="results/seedmaps/seed_summary.txt",
        help="Path to seed–track summary txt (optional but recommended).",
    )
    ap.add_argument(
        "--seed-analysis",
        default=None,
        help="Optional per-hour seed analysis txt (from analyze_seeds.py).",
    )
    ap.add_argument(
        "--alerts-dir",
        default="results/alerts",
        help="Folder containing alerts_* CSVs (for presence snapshot).",
    )
    ap.add_argument(
        "--viability-targets",
        default="data/grid_train_gse_panel_targets.parquet",
        help="Panel with y_viable/t_to_storm_min_h for conversion snapshot (optional).",
    )
    ap.add_argument(
        "--viability-metrics",
        default="models/viability_model_metrics.json",
        help="Viability model metrics JSON (optional).",
    )
    ap.add_argument(
        "--viability-horizons",
        default="24,48,72,120",
        help="Comma-separated horizons (hours) for conversion snapshot.",
    )
    ap.add_argument(
        "--extras",
        default="{}",
        help='JSON dict of extra label->path entries to surface in the report.',
    )
    # Compatibility: accept chunk hints without using them.
    ap.add_argument("--chunk-rows", type=int, default=None, help="Ignored; accepted for pipeline compatibility.")
    ap.add_argument("--chunksize", type=int, default=None, help="Alias for --chunk-rows (ignored).")
    ap.add_argument("--parquet-rows", type=int, default=None, help="Ignored; accepted for pipeline compatibility.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_slug = slugify_name(args.run_name)
    report_txt = out_dir / f"{run_slug}_report.txt"

    # ---- Seed / track summaries ----
    seed_sum_txt = read_text_safe(Path(args.seed_summary))
    seed_analysis_txt = read_text_safe(Path(args.seed_analysis)) if args.seed_analysis else ""

    # ---- Alerts snapshot ----
    alerts_dir = Path(args.alerts_dir)
    alert_rows_sampled = 0
    alert_files = 0
    if alerts_dir.exists():
        for p in alerts_dir.glob("alerts_*.csv*"):
            alert_files += 1
            try:
                # Just sample a few rows to prove the file is readable / non-empty
                c = pd.read_csv(p, nrows=1000)
                alert_rows_sampled += len(c)
            except Exception:
                # We don't fail the report if one file is unreadable
                continue

    # ---- Viability conversion snapshot (optional) ----
    conv_txt = ""
    horizons = []
    try:
        horizons = [int(h.strip()) for h in str(args.viability_horizons).split(",") if h.strip()]
    except Exception:
        horizons = [24, 48, 72, 120]

    conv_path = Path(args.viability_targets)
    if exists_nonempty(conv_path):
        try:
            cols = ["t_to_storm_min_h", "y_viable"]
            df = pd.read_parquet(conv_path, columns=cols)
            rows = []
            for h in horizons:
                mask = pd.to_numeric(df["t_to_storm_min_h"], errors="coerce").le(h)
                mask &= df["t_to_storm_min_h"].notna()
                subset = df.loc[mask]
                total = len(subset)
                pos = subset["y_viable"].sum() if "y_viable" in subset else 0
                rate = (subset["y_viable"].mean() if total > 0 else np.nan) if "y_viable" in subset else np.nan
                rows.append({"horizon_h": h, "rows": int(total), "y_viable_mean": float(rate) if pd.notna(rate) else np.nan, "positives": float(pos)})
            conv = pd.DataFrame(rows)
            conv_txt = conv.to_string(index=False)
        except Exception:
            conv_txt = ""

    # ---- Viability metrics snapshot (optional) ----
    metrics_txt = ""
    metrics_path = Path(args.viability_metrics)
    if exists_nonempty(metrics_path):
        try:
            m = json.loads(metrics_path.read_text())
            # Pick a few commonly logged metrics if present
            keys = ["roc_auc", "pr_auc", "brier", "opt_threshold", "neg_pos_ratio", "sample_frac"]
            lines = []
            for k in keys:
                if k in m:
                    lines.append(f"{k}: {m[k]}")
            if lines:
                metrics_txt = "\n".join(lines)
        except Exception:
            metrics_txt = ""

    # ---- Extras ----
    try:
        extras = json.loads(args.extras) if args.extras else {}
        if not isinstance(extras, dict):
            extras = {}
    except Exception:
        extras = {}

    # ---- Write report ----
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(f"Run Report — {args.run_name}\n")
        f.write("=" * 72 + "\n\n")

        if seed_sum_txt:
            f.write("Seed–Track Summary\n")
            f.write("-" * 72 + "\n")
            f.write(seed_sum_txt.strip() + "\n\n")

        if seed_analysis_txt:
            f.write("Seed Analysis Notes\n")
            f.write("-" * 72 + "\n")
            f.write(seed_analysis_txt.strip() + "\n\n")

        # Alerts
        f.write("Alerts snapshot\n")
        f.write("-" * 72 + "\n")
        f.write(f"Alerts directory : {alerts_dir}\n")
        f.write(f"Alert files seen : {alert_files}\n")
        f.write(
            "Sampled rows read: "
            f"{alert_rows_sampled} (for basic presence/health check only)\n\n"
        )

        # Viability conversion snapshot
        if conv_txt:
            f.write("Viability conversion snapshot (t_to_storm_min_h)\n")
            f.write("-" * 72 + "\n")
            f.write(conv_txt.strip() + "\n\n")

        # Viability metrics
        if metrics_txt:
            f.write("Viability model metrics\n")
            f.write("-" * 72 + "\n")
            f.write(metrics_txt.strip() + "\n\n")

        # Extras
        if extras:
            f.write("Extra artifacts\n")
            f.write("-" * 72 + "\n")
            for k, v in extras.items():
                f.write(f"{k}: {v}\n")

    print(f"[summary] wrote {report_txt}")


if __name__ == "__main__":
    main()

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
import sys
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


def read_table_any(path: Path, columns=None, nrows=None):
    """
    Lightweight CSV/Parquet reader with safe fallbacks.
    """
    try:
        if path.suffix.lower() in {".parquet", ".pq"}:
            df = pd.read_parquet(path, columns=columns)
            return df.head(nrows) if nrows else df
        return pd.read_csv(path, usecols=columns, nrows=nrows)
    except Exception:
        return None


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
        "--conversion-csv",
        "--include-conversion",
        dest="conversion_csv",
        default=None,
        help="Optional conversion/proto-outcomes CSV to embed in the summary.",
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
        "--viability-thresholds",
        default="results/sweeps/viability_best_thresholds.csv",
        help="Per-lead viability thresholds CSV (optional).",
    )
    ap.add_argument(
        "--seed-union",
        default=None,
        help="Seed union file (csv/parquet) to derive slow-tick diagnostics (optional).",
    )
    ap.add_argument(
        "--viability-horizons",
        default="24,48,72,120",
        help="Comma-separated horizons (hours) for conversion snapshot.",
    )
    ap.add_argument(
        "--ibtracs",
        default=None,
        help="Optional IBTrACS subset (not required; surfaced for reference).",
    )
    ap.add_argument(
        "--ibtracs-area",
        default=None,
        help='Optional AOI string used elsewhere (surfaced for reference).',
    )
    ap.add_argument(
        "--ibtracs-normalize-lon",
        choices=["none", "-180..180", "0..360"],
        default="-180..180",
        help="Lon frame used elsewhere (surfaced for reference).",
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
    argv = []
    skip = False
    raw = sys.argv[1:]
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
    args = ap.parse_args(argv)

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

    # ---- IBTrACS presence snapshot ----
    ibtracs_txt = ""
    if args.ibtracs:
        ib_path = Path(args.ibtracs)
        if exists_nonempty(ib_path):
            size_mb = ib_path.stat().st_size / 1e6
            ibtracs_txt = f"{ib_path} (size ~{size_mb:.1f} MB)"
        else:
            ibtracs_txt = f"{ib_path} (missing or empty)"

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
            df = read_table_any(conv_path, columns=cols)
            if df is None:
                df = pd.DataFrame(columns=cols)
            lead_vals = pd.to_numeric(df["t_to_storm_min_h"], errors="coerce")
            yv = pd.to_numeric(df.get("y_viable", pd.Series(dtype=float)), errors="coerce")
            rows = []
            start = 0.0
            for h in horizons:
                mask = lead_vals.gt(start) & lead_vals.le(float(h))
                subset = yv.loc[mask]
                total = int(mask.sum())
                pos = float(subset.sum()) if total > 0 else 0.0
                rate = float(subset.mean()) if total > 0 else np.nan
                rows.append(
                    {
                        "horizon": f"({start},{h}]",
                        "rows": total,
                        "y_viable_mean": rate if pd.notna(rate) else np.nan,
                        "positives": pos,
                    }
                )
                start = float(h)
            # trailing bucket > last horizon
            tail_mask = lead_vals.gt(start)
            if tail_mask.any():
                subset = yv.loc[tail_mask]
                total = int(tail_mask.sum())
                pos = float(subset.sum()) if total > 0 else 0.0
                rate = float(subset.mean()) if total > 0 else np.nan
                rows.append(
                    {
                        "horizon": f"> {start}",
                        "rows": total,
                        "y_viable_mean": rate if pd.notna(rate) else np.nan,
                        "positives": pos,
                    }
                )
            conv = pd.DataFrame(rows)
            conv_txt = conv.to_string(index=False)
        except Exception:
            conv_txt = ""

    # ---- Viability threshold snapshot (optional) ----
    thr_txt = ""
    thr_path = Path(args.viability_thresholds)
    if exists_nonempty(thr_path):
        tbl = read_table_any(thr_path)
        if tbl is None:
            tbl = read_table_any(thr_path, nrows=50)
        if tbl is not None and not tbl.empty:
            # allow various lead column spellings
            for cand in ["lead_h", "lead", "lead_hours"]:
                if cand in tbl.columns:
                    tbl = tbl.rename(columns={cand: "lead_h"})
                    break
            keep = [c for c in ["lead_h", "thr_Fbeta", "thr_fbeta", "thr_F1", "Fbeta", "F1", "precision", "recall", "coverage"] if c in tbl.columns]
            if keep:
                thr_txt = tbl[keep].head(12).to_string(index=False)

    # ---- Viability metrics snapshot (optional) ----
    metrics_txt = ""
    metrics_path = Path(args.viability_metrics)
    if exists_nonempty(metrics_path):
        try:
            m = json.loads(metrics_path.read_text())
            lines = []
            if isinstance(m, dict):
                for split in ("train", "val", "overall"):
                    if split in m and isinstance(m[split], dict):
                        part = m[split]
                        auc = part.get("roc_auc")
                        ap = part.get("avg_precision") or part.get("pr_auc")
                        brier = part.get("brier")
                        thr = part.get("opt_threshold") or part.get("thr_Fbeta")
                        pieces = [f"{split.title():<6}"]
                        if auc is not None:
                            pieces.append(f"AUC={float(auc):.3f}")
                        if ap is not None:
                            pieces.append(f"PRAUC={float(ap):.3f}")
                        if brier is not None:
                            pieces.append(f"Brier={float(brier):.4f}")
                        if thr is not None:
                            pieces.append(f"thr={thr}")
                        if len(pieces) > 1:
                            lines.append("  " + "  ".join(pieces))
                if not lines:
                    base = m.get("overall", m)
                    for k, v in base.items():
                        if isinstance(v, (int, float, str)) and len(lines) < 8:
                            lines.append(f"{k}: {v}")
            if lines:
                metrics_txt = "\n".join(lines)
        except Exception:
            metrics_txt = ""

    # ---- Slow-tick diagnostics (optional) ----
    slowtick_txt = ""
    union_path = None
    if args.seed_union:
        union_path = Path(args.seed_union)
    elif args.run_name:
        union_path = Path(f"results/seedmaps/{args.run_name}_union_byhour.parquet")
    if union_path and exists_nonempty(union_path):
        try:
            seeds = read_table_any(union_path)
            if seeds is None:
                seeds = read_table_any(union_path, nrows=1_000_000)
            if seeds is not None and not seeds.empty:
                prob_col = None
                for c in ("prob_max", "prob_viable", "prob"):
                    if c in seeds.columns:
                        prob_col = c
                        break
                if prob_col is None:
                    prob_col = "prob_max"
                    seeds[prob_col] = np.nan
                hi = seeds
                if prob_col in seeds:
                    hi = seeds[pd.to_numeric(seeds[prob_col], errors="coerce") >= 0.5]
                has_slow = {"slow_cos", "slow_sin"}.issubset(seeds.columns)
                if has_slow and not hi.empty:
                    phase = np.arctan2(
                        pd.to_numeric(hi["slow_sin"], errors="coerce"),
                        pd.to_numeric(hi["slow_cos"], errors="coerce"),
                    )
                    phase_hours = (phase % (2 * np.pi)) * 24.0 / (2 * np.pi)
                    bins = np.arange(0, 25, 3)
                    hist, _ = np.histogram(phase_hours, bins=bins)
                    slow_lines = [
                        f"High-probability seeds (prob >=0.5): {len(hi):,}",
                        f"Mean slow-phase (h): {float(np.nanmean(phase_hours)):.2f}",
                        f"Std slow-phase (h): {float(np.nanstd(phase_hours)):.2f}",
                        "Phase distribution (3h bins):",
                    ]
                    for k in range(len(bins) - 1):
                        slow_lines.append(f"  {bins[k]:2.0f}-{bins[k+1]:2.0f} h : {int(hist[k]):7d}")
                    slowtick_txt = "\n".join(slow_lines)
        except Exception:
            slowtick_txt = ""

    # ---- Proto outcomes / conversion CSV (optional) ----
    proto_txt = ""
    if args.conversion_csv:
        proto_path = Path(args.conversion_csv)
        if exists_nonempty(proto_path):
            tbl = read_table_any(proto_path, nrows=500)
            if tbl is not None and not tbl.empty:
                proto_txt = tbl.head(20).to_string(index=False)

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

        if ibtracs_txt:
            f.write("IBTrACS reference\n")
            f.write("-" * 72 + "\n")
            f.write(f"{ibtracs_txt}\n")
            if args.ibtracs_area:
                f.write(f"AOI: {args.ibtracs_area}\n")
            if args.ibtracs_normalize_lon:
                f.write(f"Lon frame: {args.ibtracs_normalize_lon}\n")
            f.write("\n")

        # Viability conversion snapshot
        if conv_txt:
            f.write("Viability conversion snapshot (t_to_storm_min_h)\n")
            f.write("-" * 72 + "\n")
            f.write(conv_txt.strip() + "\n\n")

        # Viability thresholds
        if thr_txt:
            f.write("Viability thresholds (sweep)\n")
            f.write("-" * 72 + "\n")
            f.write(thr_txt.strip() + "\n\n")

        # Viability metrics
        if metrics_txt:
            f.write("Viability model metrics\n")
            f.write("-" * 72 + "\n")
            f.write(metrics_txt.strip() + "\n\n")

        if proto_txt:
            f.write("Proto outcomes / conversion rates\n")
            f.write("-" * 72 + "\n")
            f.write(proto_txt.strip() + "\n\n")

        if slowtick_txt:
            f.write("Slow-tick diagnostics\n")
            f.write("-" * 72 + "\n")
            f.write(slowtick_txt.strip() + "\n\n")

        # Extras
        if extras:
            f.write("Extra artifacts\n")
            f.write("-" * 72 + "\n")
            for k, v in extras.items():
                f.write(f"{k}: {v}\n")

    print(f"[summary] wrote {report_txt}")


if __name__ == "__main__":
    main()

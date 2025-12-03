#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pick_best_from_sweep.py

Select the best-performing parameter sets from a sweep summary
(e.g. results/sweep_summary.csv) under configurable constraints.

Works with:
  • sweep_runner.py outputs:
        lead, thr, persist, neighbors, quantile,
        F1, Precision, Recall, Coverage, AUC, PRAUC, Brier, ...
  • sweep_gate_runner.py outputs:
        lead, tb, tr,
        F1, Precision, Recall, Coverage, AUC, PRAUC, Brier, ...

New-style usage (for run_pipeline + sweep_manager)
--------------------------------------------------
# In YAML:
sweep:
  enabled: true
  steps:
    - {mode: run,  out: results/sweeps/demo/sweep_summary.csv}
    - {mode: pick, sweep_dir: results/sweeps/demo, metric: F1,
       out: results/sweeps/demo/best.csv}

Direct CLI example
------------------
python pick_best_from_sweep.py \
  --csv results/sweeps/demo/sweep_summary.csv \
  --metric F1 \
  --coverage-max 0.25 \
  --recall-min 0.20 \
  --prec-min 0.10 \
  --topn 5 \
  --out results/sweeps/demo/best.csv

Back-compat:
  • --save is an alias for --out
  • You can still call it with only --csv/--save like before.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


# ---------------- I/O helpers ----------------

def read_any(path: Path) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith((".parquet", ".pq", ".pqt")):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False, compression="infer")


def to_num(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def resolve_sweep_file(csv_arg: str | None, sweep_dir_arg: str | None) -> Path:
    """
    Resolve which sweep summary file to use.

    Priority:
      1) --csv (explicit file)
      2) --sweep-dir:
           - if file -> use directly
           - if dir  -> try sweep*.csv, *.csv
      3) default: results/sweep_summary.csv
    """
    if csv_arg:
        p = Path(csv_arg)
        if not p.exists():
            raise SystemExit(f"[pick] --csv path not found: {p}")
        return p

    if sweep_dir_arg:
        p = Path(sweep_dir_arg)
        if p.is_file():
            return p
        if p.is_dir():
            # Prefer sweep-like names
            candidates = list(p.glob("sweep*.csv"))
            if not candidates:
                candidates = list(p.glob("*.csv"))
            if not candidates:
                raise SystemExit(f"[pick] No CSV files found in sweep_dir: {p}")
            # Stable ordering
            candidates = sorted(candidates)
            print(f"[pick] sweep_dir={p} -> using {candidates[0]}")
            return candidates[0]
        raise SystemExit(f"[pick] sweep_dir is neither file nor directory: {p}")

    # Fallback default
    p = Path("results/sweep_summary.csv")
    if not p.exists():
        raise SystemExit(f"[pick] No input specified and default not found: {p}")
    return p


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(
        description="Pick best-performing parameter sets from sweep summary."
    )
    ap.add_argument(
        "--csv",
        default=None,
        help="Sweep summary CSV/Parquet path (overrides --sweep-dir).",
    )
    ap.add_argument(
        "--sweep-dir",
        default=None,
        help="Directory or file produced by sweep_runner/sweep_gate_runner.",
    )
    ap.add_argument(
        "--metric",
        default=None,
        help="Metric column to rank by (e.g. F1_fix, F1, PRAUC). "
             "Default: F1_fix if present, else F1.",
    )
    ap.add_argument(
        "--coverage-max",
        type=float,
        default=0.25,
        help="Max allowed coverage fraction.",
    )
    ap.add_argument(
        "--recall-min",
        type=float,
        default=0.20,
        help="Min required recall.",
    )
    ap.add_argument(
        "--prec-min",
        type=float,
        default=0.10,
        help="Min required precision.",
    )
    ap.add_argument(
        "--topn",
        type=int,
        default=5,
        help="Top-N per lead to show.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output CSV with best-per-lead thresholds.",
    )
    ap.add_argument(
        "--save",
        default=None,
        help="(Back-compat) Alias for --out.",
    )

    args = ap.parse_args()

    # Resolve input path
    csv_path = resolve_sweep_file(args.csv, args.sweep_dir)
    print(f"[pick] Loading sweep summary from: {csv_path}")
    df = read_any(csv_path)
    print(f"[pick] Loaded {len(df):,} rows")

    # Normalise numeric columns (support both run + gate sweeps)
    num_cols = [
        "lead",
        "thr", "persist", "neighbors", "quantile",   # sweep_runner
        "tb", "tr",                                  # sweep_gate_runner
        "Precision", "Recall", "Coverage",
        "AUC", "PRAUC", "Brier", "F1",
    ]
    df = to_num(df, num_cols)

    # Recompute F1 from P & R if available
    if {"Precision", "Recall"}.issubset(df.columns):
        pr = df["Precision"].clip(0, 1).fillna(0.0)
        rc = df["Recall"].clip(0, 1).fillna(0.0)
        df["F1_fix"] = (2 * pr * rc) / (pr + rc + 1e-9)
    else:
        df["F1_fix"] = df.get("F1", np.nan)

    # Pick ranking metric
    if args.metric:
        metric_col = args.metric
        if metric_col not in df.columns:
            raise SystemExit(f"[pick] Requested metric '{metric_col}' not found in columns: {list(df.columns)}")
    else:
        if "F1_fix" in df.columns and df["F1_fix"].notna().any():
            metric_col = "F1_fix"
        elif "F1" in df.columns:
            metric_col = "F1"
        else:
            raise SystemExit("[pick] No F1/F1_fix metric available to rank by.")

    print(f"[pick] Ranking rows by metric: {metric_col}")

    # Apply constraints where columns exist
    for col in ("Coverage", "Recall", "Precision"):
        if col not in df.columns:
            raise SystemExit(f"[pick] Sweep summary missing required column '{col}'.")

    mask = (
        (df["Coverage"] <= args.coverage_max) &
        (df["Recall"]   >= args.recall_min) &
        (df["Precision"] >= args.prec_min)
    )
    ok = df[mask].copy()

    if ok.empty:
        print("[pick] No rows met strict constraints.")
        print("[pick] Relaxing to Coverage<=0.35 & Recall>=0.10 …")
        mask_relaxed = (df["Coverage"] <= 0.35) & (df["Recall"] >= 0.10)
        ok = df[mask_relaxed].copy()
        if ok.empty:
            print("[pick] Still empty; using all rows.")
            ok = df.copy()

    # Sort by lead then metric, recall, precision
    ok = ok.sort_values(
        ["lead", metric_col, "Recall", "Precision"],
        ascending=[True, False, False, False],
    )

    # Build list of columns to display / save
    param_cols = [
        "lead",
        "thr", "persist", "neighbors", "quantile",  # run sweeps
        "tb", "tr",                                 # gate sweeps
    ]
    metric_cols = ["F1_fix", "F1", "Precision", "Recall", "Coverage", "AUC", "PRAUC", "Brier"]

    keep_cols = [c for c in param_cols + metric_cols if c in ok.columns]

    # Best per lead
    best = ok.groupby("lead", as_index=False).head(1)[keep_cols]

    # For compatibility with grid_score thresholds_csv, add thr_Fbeta if we have thr
    if "thr" in best.columns and "thr_Fbeta" not in best.columns:
        best = best.copy()
        best["thr_Fbeta"] = best["thr"]

    print("\n== Best per lead (constraints applied) ==\n")
    if not best.empty:
        print(best.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    else:
        print("[pick] No best rows to display (this should not usually happen).")

    # Top-N per lead for context
    print(f"\n== Top-{args.topn} per lead ==\n")
    topn = ok.groupby("lead", as_index=False).head(args.topn)[keep_cols]
    if not topn.empty:
        print(topn.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    else:
        print("[pick] No top-N rows to display.")

    # Decide output path
    out_path = args.out or args.save or "results/best_fbeta_thresholds.csv"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best.to_csv(out_path, index=False)
    print(f"\n[pick] Saved best-per-lead table -> {out_path}")

    # No more suggested apply_thresholds.py commands — in the refactored pipeline
    # you typically use this CSV as thresholds_csv in grid_score.py (thr_col=thr_Fbeta).


if __name__ == "__main__":
    main()
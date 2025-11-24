#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pick_best_from_sweep.py

Selects the best-performing parameter sets from a sweep summary
(e.g. results/sweep_summary.csv) under configurable constraints.

Features
--------
- Accepts CSV or Parquet (auto-detects)
- Auto-recomputes F1 from Precision/Recall if needed
- Filters to avoid degenerate low-recall or high-coverage rows
- Reports best per-lead and top-N table
- Optionally writes best rows → results/best_fbeta_thresholds.csv
  for direct use in the pipeline

Example
-------
python pick_best_from_sweep.py \
  --csv results/sweep_summary.csv \
  --coverage-max 0.25 \
  --recall-min 0.20 \
  --prec-min 0.10 \
  --topn 5 \
  --save best_fbeta_thresholds.csv
"""

import argparse, numpy as np, pandas as pd
from pathlib import Path

# ---------- helpers ----------

def read_any(path):
    p = str(path).lower()
    if p.endswith((".parquet", ".pq", ".pqt")):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)

def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Pick best-performing parameter sets from sweep summary.")
    ap.add_argument("--csv", default="results/sweep_summary.csv", help="Sweep summary CSV/Parquet path.")
    ap.add_argument("--coverage-max", type=float, default=0.25, help="Max allowed coverage fraction.")
    ap.add_argument("--recall-min",   type=float, default=0.20, help="Min required recall.")
    ap.add_argument("--prec-min",     type=float, default=0.10, help="Min required precision.")
    ap.add_argument("--topn",         type=int, default=5, help="Top-N per lead to show.")
    ap.add_argument("--save", default="results/best_fbeta_thresholds.csv",
                    help="Optional CSV to save best-per-lead thresholds.")
    args = ap.parse_args()

    df = read_any(args.csv)
    print(f"Loaded {len(df):,} rows from {args.csv}")

    num_cols = ["lead","thr","persist","neighbors","quantile",
                "Precision","Recall","Coverage","AUC","PRAUC","Brier","F1"]
    df = to_num(df, num_cols)

    # recompute F1 from P & R
    if {"Precision","Recall"}.issubset(df.columns):
        pr = df["Precision"].clip(0,1).fillna(0)
        rc = df["Recall"].clip(0,1).fillna(0)
        df["F1_fix"] = (2*pr*rc) / (pr + rc + 1e-9)
    else:
        df["F1_fix"] = df.get("F1", np.nan)

    # apply constraints
    mask = (
        (df["Coverage"] <= args.coverage_max) &
        (df["Recall"]   >= args.recall_min) &
        (df["Precision"]>= args.prec_min)
    )
    ok = df[mask].copy()

    if ok.empty:
        print("No rows met strict constraints. Relaxing to Coverage<=0.35 & Recall>=0.10 …")
        ok = df[(df["Coverage"] <= 0.35) & (df["Recall"] >= 0.10)].copy()
        if ok.empty:
            print("Still empty; using all rows.")
            ok = df.copy()

    keep_cols = ["lead","thr","persist","neighbors","quantile",
                 "F1_fix","Precision","Recall","Coverage","AUC","PRAUC","Brier"]
    ok = ok.sort_values(["lead","F1_fix","Recall","Precision"],
                        ascending=[True, False, False, False])

    # best per lead
    best = ok.groupby("lead", as_index=False).head(1)[keep_cols]
    print("\n== Best per lead (constraints applied) ==\n")
    print(best.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # top-N per lead for context
    print(f"\n== Top-{args.topn} per lead ==\n")
    topn = ok.groupby("lead", as_index=False).head(args.topn)[keep_cols]
    print(topn.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Save best table for pipeline threshold search
    out_path = Path(args.save)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best.to_csv(out_path, index=False)
    print(f"\nSaved best-per-lead thresholds → {out_path}")

    # Generate suggested apply commands
    print("\n== Suggested apply commands ==")
    for _, r in best.iterrows():
        cmd = (
            f"python apply_thresholds.py "
            f"--labelled data/grid_labelled_FMA_gka.csv.gz "
            f"--model models/grid_logit_cal.pkl "
            f"--lead-hours {int(r.lead)} "
            f"--thr {r.thr:.3f} "
            f"--out results/alerts_best_lead{int(r.lead)}.csv.gz"
        )
        print(" ", cmd)

if __name__ == "__main__":
    main()
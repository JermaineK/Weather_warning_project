#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hourly_rollup.py

Roll up alert scores and flags to hourly aggregates.

Features
- Accepts CSV(.gz) or Parquet
- Floors timestamps to exact hours (handles slightly off-minute stamps)
- Aggregates one or more risk columns (auto-detects 'risk' columns if not given)
- Optionally counts a final alert flag column (e.g., alert_final)
- Adds per-hour cell counts and (lat,lon) uniqueness
- Writes CSV(.gz) or Parquet based on --out extension

Examples
--------
python hourly_rollup.py \
  --alerts results/alerts/alerts_gka_fma_demo_lead72_thr0.0435.csv.gz \
  --flag-col alert_final \
  --risk-cols risk_final,risk \
  --out results/rollups/hourly_rollup.csv.gz
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------- I/O helpers ----------------

def read_any(path, parse_dates=None, usecols=None):
    p = str(path).lower()
    if p.endswith((".parquet", ".pq", ".pqt")):
        return pd.read_parquet(path, columns=usecols if usecols else None)
    return pd.read_csv(path,
                       low_memory=False,
                       parse_dates=parse_dates if parse_dates else None,
                       usecols=usecols if usecols else None)

def write_any(path, df):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    p = str(path).lower()
    if p.endswith((".parquet", ".pq", ".pqt")):
        df.to_parquet(path, index=False)
    else:
        comp = "gzip" if p.endswith(".gz") else "infer"
        df.to_csv(path, index=False, compression=comp, date_format="%Y-%m-%d %H:%M:%S")

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="Hourly roll-up of alert risks/flags.")
    ap.add_argument("--alerts", required=True, help="CSV(.gz)/Parquet with cols time,lat,lon,+risk/flags")
    ap.add_argument("--flag-col", default="alert_final",
                    help="Final alert flag to count (e.g., alert_final). If missing, skip.")
    ap.add_argument("--risk-cols", default=None,
                    help="Comma-separated risk columns to aggregate (default: auto-detect 'risk*').")
    ap.add_argument("--out", required=True, help="Output CSV(.gz)/Parquet")
    args = ap.parse_args()

    # Load minimal columns first; expand later for risks/flag
    df = read_any(args.alerts, parse_dates=["time"])
    if "time" not in df.columns:
        raise ValueError("Input must contain a 'time' column.")
    # Lat/lon are optional for rollup, but help report cell counts
    has_lat = "lat" in df.columns
    has_lon = "lon" in df.columns

    # Normalize time -> hour
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True).dt.tz_localize(None)
    df["_hour"] = df["time"].dt.floor("H")

    # Pick risk columns
    risk_cols = []
    if args.risk_cols:
        risk_cols = [c.strip() for c in args.risk_cols.split(",") if c.strip()]
    else:
        # auto-detect any numeric columns whose names begin with 'risk'
        for c in df.columns:
            if c.lower().startswith("risk") and pd.api.types.is_numeric_dtype(df[c]):
                risk_cols.append(c)
        # as a fallback if nothing matches, try a common default 'risk'
        if not risk_cols and "risk" in df.columns and pd.api.types.is_numeric_dtype(df["risk"]):
            risk_cols = ["risk"]

    # Keep only numeric risk columns that exist
    risk_cols = [c for c in risk_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not risk_cols:
        print("[warn] No usable risk columns found; rollup will only include counts/flag (if any).")

    # Flag column (optional)
    flag_col = args.flag_col if args.flag_col in df.columns else None
    if args.flag_col and not flag_col:
        print(f"[warn] Flag column '{args.flag_col}' not found; skipping flag counts.")

    # Build aggregation dict
    aggs = {}
    for rc in risk_cols:
        aggs[rc] = ["min", "median", "max", "mean"]
    if flag_col:
        aggs[flag_col] = ["sum"]

    # Always add counts
    # - rows_per_hour: number of rows in that hour
    # - cells_per_hour: unique (lat,lon) in that hour if available
    group = df.groupby("_hour", sort=False)

    roll = group.agg(aggs) if aggs else group.size().to_frame("rows_per_hour")

    # Flatten MultiIndex columns if needed
    if aggs:
        roll.columns = ["_".join([c for c in tup if c]) for tup in roll.columns.to_flat_index()]

    # Counts
    roll["rows_per_hour"] = group.size().to_numpy()
    if has_lat and has_lon:
        # unique cell count per hour
        roll["cells_per_hour"] = group.apply(lambda g: pd.DataFrame({"lat": g["lat"], "lon": g["lon"]})
                                             .dropna()
                                             .drop_duplicates()
                                             .shape[0]).to_numpy()
    else:
        roll["cells_per_hour"] = np.nan

    # Reset index to expose time column
    roll = roll.reset_index().rename(columns={"_hour": "time"})

    # Order columns: time, counts, flag sums, risks...
    col_order = ["time", "rows_per_hour", "cells_per_hour"]
    if flag_col:
        col_order += [f"{flag_col}_sum"]
    # then risk stats grouped by each risk col
    for rc in risk_cols:
        col_order += [f"{rc}_min", f"{rc}_median", f"{rc}_max", f"{rc}_mean"]
    # keep any leftover columns (unlikely) at the end
    col_order += [c for c in roll.columns if c not in col_order]
    roll = roll[col_order]

    write_any(args.out, roll)
    print(f"[rollup] wrote {args.out} | hours={len(roll):,} | "
          f"risk_cols={risk_cols if risk_cols else '[]'} | flag={flag_col or '(none)'}")

if __name__ == "__main__":
    main()
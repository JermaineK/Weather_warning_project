#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_synth_thermo.py
Create synthetic thermo-adjacent features from the base labelled grid.

Inputs:
  --labelled <csv.gz>   grid with columns: time (tz-aware/naive), lat, lon, ...
  --conv-window-h  <int>  hours for smoothing window (default 6)
  --persist-h      <int>  hours for persistence/max window (default 24)
  --out <csv.gz>         output file

Notes:
- Avoids deprecated groupby.apply behaviours.
- All group apply helpers return Series indexed by the group's ORIGINAL ROW INDEX.
- Final assignment always reindexes to the frame index to avoid incompatibility.
"""

import argparse
import numpy as np
import pandas as pd

# ---------------------------
# Utilities
# ---------------------------

def coerce_time_naive_utc(df: pd.DataFrame, col: str = "time") -> pd.DataFrame:
    """Ensure time is timezone-naive UTC-like (remove tz) and dtype datetime64[ns]."""
    t = pd.to_datetime(df[col], errors="coerce")
    # If time is tz-aware, convert to UTC then drop tz
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.copy()
    df[col] = t.astype("datetime64[ns]")
    return df

def try_group_apply(gb, func, **kwargs) -> pd.Series:
    """
    Apply a function per group that returns a Series aligned to the GROUP'S ORIGINAL
    ROW INDEX. Concatenate (group_keys=False on the groupby call) and then
    return a flat Series indexed by the original DataFrame's row index order.
    """
    out = gb.apply(lambda g: func(g, **kwargs))
    # out should already be a flat Series with the original row index for all groups.
    if not isinstance(out, pd.Series):
        out = pd.Series(out)
    # Safety: sort by index to match the parent frame order expectation
    return out.sort_index()

# ---------------------------
# Rolling helpers (index-safe)
# ---------------------------

def _rolling_time_series(values: pd.Series, times: pd.Series, hours: int, agg: str, shift_hours: int = 0) -> pd.Series:
    """
    Generic time-based rolling on a single column for one group.
    Returns a Series indexed by the group's ORIGINAL ROW INDEX (values.index).
    - values: numeric Series (group) with original row index.
    - times:  datetime Series (group) with original row index.
    - hours:  window size in hours.
    - agg:    'mean' or 'max'
    - shift_hours: optional positive shift (hours) applied AFTER rolling.
    """
    # Ensure proper dtypes and sort by time
    t = pd.to_datetime(times, errors="coerce")
    order = np.argsort(t.values)
    idx_sorted = values.index.take(order)
    t_sorted = pd.DatetimeIndex(t.iloc[order].values)
    v_sorted = pd.Series(pd.to_numeric(values.iloc[order], errors="coerce").values, index=t_sorted)

    # Rolling
    win = v_sorted.rolling(f"{hours}h", min_periods=1)
    if agg == "mean":
        r_sorted = win.mean()
    elif agg == "max":
        r_sorted = win.max()
    else:
        raise ValueError("agg must be 'mean' or 'max'")

    if shift_hours and shift_hours > 0:
        r_sorted = r_sorted.shift(freq=pd.Timedelta(hours=shift_hours))

    # Map back to original row order (by time lookup on unsorted times)
    r_unsorted = r_sorted.reindex(t)  # t has original row order & length
    # Return aligned to the original row *index* (values.index)
    return pd.Series(r_unsorted.values, index=values.index)

def _roll_mean(g: pd.DataFrame, col: str, hours: int) -> pd.Series:
    return _rolling_time_series(g[col], g["time"], hours, agg="mean", shift_hours=0)

def _roll_max_shift1h(g: pd.DataFrame, col: str, hours: int) -> pd.Series:
    # often we want a "prior window" max → shift by +1h to exclude current instant
    return _rolling_time_series(g[col], g["time"], hours, agg="max", shift_hours=1)

# ---------------------------
# Feature construction
# ---------------------------

def add_synth_thermo(df: pd.DataFrame, conv_window_h: int = 6, persist_h: int = 24):
    """
    Adds synthetic thermo-like features built from existing fields typically present
    in your grid (e.g., divergence, relax, S, etc.). Designed to be safe even if some
    columns are missing — only builds from what exists.

    Returns (out_df, new_cols)
    """
    # Work on a copy, normalize/naivify time
    out = df.copy()
    out = coerce_time_naive_utc(out, "time")

    # Ensure required structural columns exist
    for c in ["lat", "lon", "time"]:
        if c not in out.columns:
            raise KeyError(f"Missing required column: {c}")

    # Candidate base columns (add more if needed/available in your file)
    base_cols = []
    # Divergence (often a decent thermo proxy)
    if "div_mean" in out.columns:
        base_cols.append("div_mean")
    elif "divergence" in out.columns:
        base_cols.append("divergence")
    # GKA invariants (already present in your data)
    for c in ["gka_kappa", "gka_tau", "gka_parity_eta", "gka_A_overlap", "gka_F", "gka_msl_nd", "gka_knee_ratio"]:
        if c in out.columns:
            base_cols.append(c)
    # Other common signals if present
    for c in ["relax", "S", "dS_dt", "drelax_dt", "msl_grad"]:
        if c in out.columns:
            base_cols.append(c)

    base_cols = list(dict.fromkeys(base_cols))  # de-dup preserve order

    added_cols = []

    if not base_cols:
        # Nothing to build from
        return out, added_cols

    # Group by grid cell. IMPORTANT: group_keys=False here (not in apply).
    gb = out.groupby(["lat", "lon"], sort=False, group_keys=False)

    # 1) Smoothed (time-conv) versions
    for col in base_cols:
        sm = try_group_apply(gb, _roll_mean, col=col, hours=conv_window_h)
        name = f"{col}_sm{conv_window_h}h"
        # Align strictly to frame index to avoid any incompatible-index errors
        out[name] = pd.to_numeric(sm, errors="coerce").reindex(out.index)
        added_cols.append(name)

    # 2) Persistence / prior-window max (exclude current hour via +1h shift in helper)
    for col in base_cols:
        pm = try_group_apply(gb, _roll_max_shift1h, col=col, hours=persist_h)
        name = f"{col}_max{persist_h}h_prev"
        out[name] = pd.to_numeric(pm, errors="coerce").reindex(out.index)
        added_cols.append(name)

    # 3) Simple thermo composites (dimensionless-ish)
    # Example: "warm-moist" like signal using gka_msl_nd and relax (if present)
    if "gka_msl_nd" in out.columns and "relax" in out.columns:
        name = "thermo_warm_moist_proxy"
        out[name] = (out["gka_msl_nd"].astype(float).clip(-5, 5) + out["relax"].astype(float).clip(-5, 5)) / 2.0
        added_cols.append(name)

    # Example: shear-suppression proxy combining dS_dt and gka_F
    if "dS_dt" in out.columns and "gka_F" in out.columns:
        name = "thermo_shear_suppress_proxy"
        out[name] = (-out["dS_dt"].astype(float).clip(-5, 5) + (1.0 - out["gka_F"].astype(float)).clip(0, 1)).clip(-10, 10)
        added_cols.append(name)

    return out, added_cols

# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Build synthetic thermo features safely.")
    ap.add_argument("--labelled", required=True, help="Input labelled grid csv.gz")
    ap.add_argument("--conv-window-h", type=int, default=6, help="Hours for smoothing window")
    ap.add_argument("--persist-h",     type=int, default=24, help="Hours for prior-window max (shifted)")
    ap.add_argument("--out",           required=True, help="Output csv.gz")
    args = ap.parse_args()

    df = pd.read_csv(args.labelled, parse_dates=["time"])
    out, added_cols = add_synth_thermo(df, conv_window_h=args.conv_window_h, persist_h=args.persist_h)

    # Summary to screen
    print("== Synth Thermo Builder ==")
    print(f"Rows: {len(out):,}")
    if added_cols:
        print(f"New columns added ({len(added_cols)}): {', '.join(added_cols)}")
        # quick percentiles for the first few
        samp = out[added_cols].sample(min(5, len(out)), random_state=42)  # tiny sanity sample
    else:
        print("No new columns added (no eligible bases found).")

    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}  | cols(+{len(added_cols)})")

if __name__ == "__main__":
    main()